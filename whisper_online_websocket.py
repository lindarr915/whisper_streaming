# import asyncio
# import logging
# from fastapi import FastAPI
# from starlette.responses import StreamingResponse

# from whisper_online import *


# from queue import Empty


# src_lan = "zh"  # source language
# tgt_lan = "zh"  # target language  -- same as source for ASR, "en" if translate task is used


# asr = FasterWhisperASR(lan, "large-v2")  # loads and wraps Whisper model
# # set options:
# # asr.set_translate_task()  # it will translate from lan into English
# # asr.use_vad()  # set using VAD 


# online = OnlineASRProcessor(tgt_lan, asr)  # create processing object


# while audio_has_not_ended:   # processing loop:
# 	a = # receive new audio chunk (and e.g. wait for min_chunk_size seconds first, ...)
# 	online.insert_audio_chunk(a)
# 	o = online.process_iter()
# 	print(o) # do something with current partial output
# # at the end of this audio processing
# o = online.finish()
# print(o)  # do something with the last output

# ``
# online.init()  # refresh if you're going to re-use the object for the next audio



# fastapi_app = FastAPI()


# class Textbot:
#     def __init__(self, model_id: str):
#         self.loop = asyncio.get_running_loop()

#         self.model_id = model_id
#         self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)


#     @fastapi_app.post("/")
#     def handle_request(self, prompt: str) -> StreamingResponse:
#         logger.info(f'Got prompt: "{prompt}"')
#         streamer = TextIteratorStreamer(
#             self.tokenizer, timeout=0, skip_prompt=True, skip_special_tokens=True
#         )
#         self.loop.run_in_executor(None, self.generate_text, prompt, streamer)
#         return StreamingResponse(
#             self.consume_streamer(streamer), media_type="text/plain"
#         )

#     def generate_text(self, prompt: str, streamer: TextIteratorStreamer):
#         input_ids = self.tokenizer([prompt], return_tensors="pt").input_ids
#         self.model.generate(input_ids, streamer=streamer, max_length=10000)

#     async def consume_streamer(self, streamer: TextIteratorStreamer):
#         while True:
#             try:
#                 for token in streamer:
#                     logger.info(f'Yielding token: "{token}"')
#                     yield token
#                 break
#             except Empty:
#                 # The streamer raises an Empty exception if the next token
#                 # hasn't been generated yet. `await` here to yield control
#                 # back to the event loop so other coroutines can run.
#                 await asyncio.sleep(0.001)
