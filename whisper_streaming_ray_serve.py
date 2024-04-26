from typing import List
from ray.serve.handle import DeploymentHandle
import asyncio
import logging
import json
import numpy as np
import soundfile
from whisper_online import *
import io
import soundfile
import numpy as np
import uuid
from ray import serve
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

import boto3



SAMPLING_RATE = 16000

logger = logging.getLogger("ray.serve")
logger.setLevel(logging.INFO)
fastapi_app = FastAPI()


@serve.deployment
@serve.ingress(fastapi_app)
class TranscriptionServer:
    def __init__(self, asr_handle: DeploymentHandle):
        self.last_end = None
        self.min_chunk = 2
        self.queue = asyncio.Queue()
        self.skip_count = 0

        self.connected_clients = {}
        self.asr_handle = asr_handle
        self.translate = boto3.client(service_name='translate', region_name='ap-northeast-1')
        

    async def handle_audio(self, client, websocket: WebSocket) -> None:
        audio: List[np.ndarray] = []

        while True:
            raw_bytes = await websocket.receive_bytes()

            sf = soundfile.SoundFile(
                io.BytesIO(raw_bytes),
                channels=1,
                endian="LITTLE",
                samplerate=SAMPLING_RATE,
                subtype="PCM_16",
                format="RAW",
            )
            audio_chunk, _ = librosa.load(sf, sr=SAMPLING_RATE)
            audio.append(audio_chunk)

            # receive all audio that is available by this time
            # blocks operation if less than self.min_chunk seconds is available
            # unblocks if connection is closed or a chunk is available
            if sum(len(data) for data in audio) >= self.min_chunk * SAMPLING_RATE:
                # self.queue.put_process_iternowait(np.concatenate(audio_chunk))
                client.insert_audio_chunk(np.concatenate(audio))
                audio.clear()
                asyncio.create_task(self.recognize(client, websocket))

    async def recognize(self, client, websocket: WebSocket) -> None:
        # while True:
        # audio_chunk = await self.queue.get()
        # logger.debug("Get item from queue!")
        # assert len(audio_chunk) > 100

        timestamped_tuple = await client.process_iter()
        if timestamped_tuple is not None:
            ws_response = self.format_output_transcript(timestamped_tuple)
            if timestamped_tuple[0] is not None:
                logger.info(f"Response object is {ws_response}")
            await websocket.send_text(str(ws_response))
        # await asyncio.sleep(0.01)

    def format_output_transcript(self, tuple):
        # output format in stdout is like:
        # 0 1720 Takhle to je
        # - the first two words are:
        #    - beg and end timestamp of the text segment, as estimated by Whisper model. The timestamps are not accurate, but they're useful anyway
        # - the next words: segment transcript

        # This function differs from whisper_online.output_transcript in the following:
        # succeeding [beg,end] intervals are not overlapping because ELITR protocol (implemented in online-text-flow events) requires it.
        # Therefore, beg, is max of previous end and current beg outputed by Whisper.
        # Usually it differs negligibly, by appx 20 ms.

        if tuple[0] is not None:
            beg, end = tuple[0] * 1000, tuple[1] * 1000
            # if self.last_end is not None:
                # beg = max(beg, self.last_end)

            # self.last_end = end

            data = {"StartTime": beg, 
                    "EndTime": end, 
                    "Transcript": tuple[2]
            }
            
            # start_time = time.time()s
            # result =  self.translate.translate_text(Text=tuple[2], SourceLanguageCode="zh", TargetLanguageCode="en")
            # duration = time.time() - start_time
            # logger.info(f"Translate duration: {duration}")
            
            # data = {"StartTime": beg, 
            #         "EndTime": end, 
            #         "Transcript": result.get('TranslatedText')
            # }

            # Use jsonify to convert the dictionary to a JSON response
            response = json.dumps(data)

            logger.info("%1.0f %1.0f %s" % (beg, end, tuple[2]))
            return response
        else:
            # logger.debug(output)
            return None

    @fastapi_app.websocket("/")
    async def handle_request(self, websocket: WebSocket, lang: str = "zh", dest_lang: str = "zh") -> None:
        tgt_lan = lang  # source language
        dst_lan = dest_lang

        await websocket.accept()
        client_id = str(uuid.uuid4())
        client = OnlineASRProcessor(
            self.asr_handle, create_tokenizer(tgt_lan), lang
        )  # create processing object
        self.connected_clients[client_id] = client

        logger.info(f"Client {client_id} connected")

        # ts_server = TranscriptionServer()

        # try:
        #     await ts_server.handle_audio(websocket)
        # except WebSocketDisconnect as e:
        #     logger.warn(f"Connection with {client_id} closed: {e}")

        try:
            await self.handle_audio(client, websocket)
        except WebSocketDisconnect as e:
            logger.warn(f"Connection with {client_id} closed: {e}")
        finally:
            del self.connected_clients[client_id]


app = TranscriptionServer.bind(FasterWhisperASR.bind())
