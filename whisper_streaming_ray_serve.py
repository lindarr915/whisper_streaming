from typing import List
from ray.serve.handle import DeploymentHandle
import asyncio
import logging
import json

import uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from ray import serve

import numpy as np

import soundfile
import io
import aiofiles

from whisper_online import *

import io
import soundfile
import numpy as np
SAMPLING_RATE = 16000

logger = logging.getLogger("ray.serve")
logger.setLevel(logging.INFO)
fastapi_app = FastAPI()


@serve.deployment
@serve.ingress(fastapi_app)
class TranscriptionServer:
    def __init__(self, asr_handle: DeploymentHandle):
        self.loop = asyncio.get_running_loop()
        self.last_end = None
        self.min_chunk = 2
        self.queue = asyncio.Queue()

        tgt_lan = "zh"  # source language
        self.asr_handle = asr_handle

        # self.asr = FasterWhisperASR(tgt_lan, "large-v2")  # loads and wraps Whisper model
        # sentence segmenter for the target language
        tokenizer = create_tokenizer(tgt_lan)
        self.online_asr_processor = OnlineASRProcessor(
            self.asr_handle, tokenizer)  # create processing object

    async def handle_audio(self, websocket: WebSocket) -> None:
        audio_chunk: List[np.ndarray] = []

        while True:
            raw_bytes = await websocket.receive_bytes()

            sf = soundfile.SoundFile(io.BytesIO(
                raw_bytes), channels=1, endian="LITTLE", samplerate=SAMPLING_RATE, subtype="PCM_16", format="RAW")
            audio, _ = librosa.load(sf, sr=SAMPLING_RATE)
            audio_chunk.append(audio)

            # receive all audio that is available by this time
            # blocks operation if less than self.min_chunk seconds is available
            # unblocks if connection is closed or a chunk is available
            if sum(len(data) for data in audio_chunk) >= self.min_chunk * SAMPLING_RATE:
                # self.queue.put_process_iternowait(np.concatenate(audio_chunk))
                asyncio.create_task(self.recognize(
                    websocket, np.concatenate(audio_chunk)))
                audio_chunk.clear()

    async def recognize(self, websocket: WebSocket, audio_data: np.ndarray) -> None:
        # while True:
        # audio_chunk = await self.queue.get()
        # logger.debug("Get item from queue!")
        # assert len(audio_chunk) > 100

        self.online_asr_processor.insert_audio_chunk(audio_data)
        result = await self.online_asr_processor.process_iter()
        response = self.format_output_transcript(result)
        if result[0] is not None:
            logger.info(f"Response object is {response}")
            await websocket.send_text(str(response))
        # await asyncio.sleep(0.01)

    def format_output_transcript(self, output):
        # output format in stdout is like:
        # 0 1720 Takhle to je
        # - the first two words are:
        #    - beg and end timestamp of the text segment, as estimated by Whisper model. The timestamps are not accurate, but they're useful anyway
        # - the next words: segment transcript

        # This function differs from whisper_online.output_transcript in the following:
        # succeeding [beg,end] intervals are not overlapping because ELITR protocol (implemented in online-text-flow events) requires it.
        # Therefore, beg, is max of previous end and current beg outputed by Whisper.
        # Usually it differs negligibly, by appx 20 ms.

        if output[0] is not None:
            beg, end = output[0] * 1000, output[1] * 1000
            if self.last_end is not None:
                beg = max(beg, self.last_end)

            self.last_end = end

            data = {
                "StartTime": beg,
                "EndTime": end,
                "Transcript": output[2]
            }

            # Use jsonify to convert the dictionary to a JSON response
            response = json.dumps(data)

            logger.info("%1.0f %1.0f %s" % (beg, end, output[2]))
            return response
        else:
            # logger.debug(output)
            return None

    @fastapi_app.websocket("/")
    async def handle_request(self, websocket: WebSocket) -> None:
        await websocket.accept()
        client_id = str(uuid.uuid4())
        logger.info(f"Client {client_id} connected")

        # ts_server = TranscriptionServer()

        # try:
        #     await ts_server.handle_audio(websocket)
        # except WebSocketDisconnect as e:
        #     logger.warn(f"Connection with {client_id} closed: {e}")

        try:
            await self.handle_audio(websocket)
        except WebSocketDisconnect:
            pass


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)

app = TranscriptionServer.bind(FasterWhisperASR.bind())
