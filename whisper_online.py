#!/usr/bin/env python3
# The module is forked from https://github.com/ufal/whisper_streaming, and modified to be a Ray Serve ready version

import sys
import numpy as np
import librosa
from functools import lru_cache
import time

import logging
from ray import serve
from ray.serve.handle import DeploymentHandle


@lru_cache
def load_audio(fname):
    audio, _ = librosa.load(fname, sr=16000)
    return audio


def load_audio_chunk(fname, beg, end):
    audio = load_audio(fname)
    beg_s = int(beg * 16000)
    end_s = int(end * 16000)
    return audio[beg_s:end_s]


logging.getLogger("faster_whisper").setLevel(logging.DEBUG)

logger = logging.getLogger("ray.serve")
logger.setLevel(logging.DEBUG)
# logger.setLevel(logging.WARNING)


# Whisper backend
class ASRBase:

    # join transcribe words with this character (" " for whisper_timestamped, "" for faster-whisper because it emits the spaces when neeeded)
    sep = " "

    def __init__(self, lang="zh", modelsize="large-v3", cache_dir=None, model_dir=None):
        self.transcribe_kargs = {}
        self.original_language = lang
        self.model = self.load_model(modelsize, cache_dir, model_dir)

    def load_model(self, modelsize, cache_dir):
        raise NotImplemented("must be implemented in the child class")

    def transcribe(self, audio, init_prompt=""):
        raise NotImplemented("must be implemented in the child class")

    def use_vad(self):
        raise NotImplemented("must be implemented in the child class")
    
    def set_language(self, lang):
        self.original_lang = lang


# @serve.deployment(ray_actor_options={"num_gpus": 1})
class HuggingFaceWhisperASR(ASRBase):
    """Use HuggingFace Transformers as the backend. Not tested or implmeneted yet."""

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        from transformers import pipeline
        import numpy as np

        self.transcriber = pipeline(
            "automatic-speech-recognition", model="openai/whisper-large-v3"
        )

    def transcribe(self, audio):
        return self.transcriber({"raw": audio})["text"]


@serve.deployment(ray_actor_options={"num_gpus": 1})
class FasterWhisperASR(ASRBase):
    """Uses faster-whisper library as the backend. Works much faster, appx 4-times (in offline mode). For GPU, it requires installation with a specific CUDNN version.

    Requires imports, if used:
        import faster_whisper
    """

    def load_model(self, modelsize=None, cache_dir=None, model_dir=None):
        from faster_whisper import WhisperModel

        if model_dir is not None:
            logger.debug(
                f"Loading whisper model from model_dir {model_dir}. modelsize and cache_dir parameters are not used."
            )
            model_size_or_path = model_dir
        elif modelsize is not None:
            model_size_or_path = modelsize
        else:
            raise ValueError("modelsize or model_dir parameter must be set")

        model = WhisperModel(
            model_size_or_path,
            device="cuda",
            compute_type="float16",
            download_root=cache_dir,
        )
        return model

    def transcribe(self, audio, init_prompt="", lang="zh"):
        # tested: beam_size=5 is faster and better than 1 (on one 200 second document from En ESIC, min chunk 0.01)

        logger.info(f"init prompt is: {init_prompt}")
        segments, info = self.model.transcribe(
            audio,
            language=lang,
            initial_prompt=init_prompt,
            beam_size=5,
            word_timestamps=True,
            condition_on_previous_text=False,
            **self.transcribe_kargs,
        )
        return list(segments)

    def use_vad(self):
        self.transcribe_kargs["vad_filter"] = True

    def set_translate_task(self):
        self.transcribe_k


class HypothesisBuffer:

    def __init__(self):
        self.commited_in_buffer = []
        self.buffer = []
        self.new = []

        self.last_commited_time = 0
        self.last_commited_word = None

    def insert(self, new, offset):
        # compare self.commited_in_buffer and new. It inserts only the words in new that extend the commited_in_buffer, it means they are roughly behind last_commited_time and new in content
        # the new tail is added to self.new

        new = [(start_time + offset, end_time + offset, t) for start_time, end_time, t in new]
        self.new = [(start_time, end_time, t) for start_time, end_time, t in new if start_time > self.last_commited_time - 0.1]

        # TODO: The following code needs to be refactored
        if len(self.new) >= 1:
            a, b, t = self.new[0]
            if abs(a - self.last_commited_time) < 1:
                if self.commited_in_buffer:
                    # it's going to search for 1, 2, ..., 5 consecutive words (n-grams) that are identical in commited and new. If they are, they're dropped.
                    cn = len(self.commited_in_buffer)
                    nn = len(self.new)
                    for i in range(1, min(min(cn, nn), 5) + 1):  # 5 is the maximum
                        c = " ".join(
                            [self.commited_in_buffer[-j][2] for j in range(1, i + 1)][
                                ::-1
                            ]
                        )
                        tail = " ".join(self.new[j - 1][2] for j in range(1, i + 1))
                        if c == tail:
                            print(f"removing last words:")
                            for j in range(i):
                                print("\t", self.new.pop(0))
                            break

    def flush(self):
        # TODO: The following code needs to be refactored
        # returns commited chunk = the longest common prefix of 2 last inserts.

        commit = []
        while self.new:
            start_time, end_time, transcription = self.new[0]

            if len(self.buffer) == 0:
                break

            if transcription == self.buffer[0][2]:
                commit.append((start_time, end_time, transcription))
                self.last_commited_word = transcription
                self.last_commited_time = end_time
                self.buffer.pop(0)
                self.new.pop(0)
            else:
                break
        self.buffer = self.new
        self.new = []
        self.commited_in_buffer.extend(commit)
        return commit

    def pop_commited(self, time):
        while self.commited_in_buffer and self.commited_in_buffer[0][1] <= time:
            self.commited_in_buffer.pop(0)

    def complete(self):
        return self.buffer


class OnlineASRProcessor:

    SAMPLING_RATE = 16000

    def __init__(self, asr_handle: DeploymentHandle, tokenizer, original_lang):
        """asr: WhisperASR object
        tokenizer: sentence tokenizer object for the target language. Must have a method *split* that behaves like the one of MosesTokenizer.
        """
        self.asr = asr_handle
        self.tokenizer = tokenizer
        self.processing = False
        self.original_lang = original_lang

        self.init()

    def init(self):
        """run this when starting or restarting processing"""
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_time_offset = 0

        self.transcript_buffer = HypothesisBuffer()
        self.commited = []
        self.last_chunked_at = 0

        self.silence_iters = 0

    def insert_audio_chunk(self, audio):
        self.audio_buffer = np.append(self.audio_buffer, audio)

    def prompt(self):
        """Returns a tuple: (prompt, context), where "prompt" is a 200-character suffix of commited text that is inside of the scrolled away part of audio buffer.
        "context" is the commited text that is inside the audio buffer. It is transcribed again and skipped. It is returned only for debugging and logging reasons.
        """
        # TODO: The following code needs to be refactored
        k = max(0, len(self.commited) - 1)
        while k > 0 and self.commited[k - 1][1] > self.last_chunked_at:
            k -= 1

        p = self.commited[:k]
        p = [t for _, _, t in p]
        prompt = []
        l = 0
        while p and l <= 100:  # 200 characters prompt size
            x = p.pop(-1)
            l += len(x) + 1
            prompt.append(x)
        non_prompt = self.commited[k:]
        return "".join(prompt[::-1]), "".join(t for _, _, t in non_prompt)

    def ts_words(self, segments):
        o = []
        for segment in segments:
            for word in segment.words:
                # not stripping the spaces -- should not be merged with them!
                w = word.word
                t = (word.start, word.end, w)
                o.append(t)
        return o

    async def process_iter(self):
        """Runs on the current audio buffer.
        Returns: a tuple (beg_timestamp, end_timestamp, "text"), or (None, None, "").
        The non-emty text is confirmed (commited) partial transcript.
        """

        prompt, non_prompt = self.prompt()
        logger.debug(f"PROMPT: {prompt}")
        logger.debug(f"CONTEXT: {non_prompt}")
        logger.info(
            f"transcribing {len(self.audio_buffer)/self.SAMPLING_RATE:2.2f} seconds from {self.buffer_time_offset:2.2f}"
        )

        if self.processing:
            logger.warning(
                "Processing is already in progress. Skipping this iteration."
            )
            return None

        self.processing = True
        res = await self.asr.transcribe.remote(self.audio_buffer, init_prompt=prompt, lang=self.original_lang)

        # transform to [(beg,end,"word1"), ...]
        tsw = self.ts_words(res)

        self.transcript_buffer.insert(tsw, self.buffer_time_offset)
        o = self.transcript_buffer.flush()
        self.commited.extend(o)
        logger.debug(f">>>>COMPLETE NOW: {self.to_flush(o)}")
        logger.debug(f"INCOMPLETE: {self.to_flush(self.transcript_buffer.complete())}")

        # there is a newly confirmed text
        if o:
            # we trim all the completed sentences from the audio buffer
            self.chunk_completed_sentence()

        #     ...segments could be considered
        #     self.chunk_completed_segment(res)

        #            self.silence_iters = 0

        # this was an attempt to trim silence/non-linguistic noise detected by the fact that Whisper doesn't transcribe anything for 3-times in a row.
        # It seemed not working better, or needs to be debugged.

        #        elif self.transcript_buffer.complete():
        #            self.silence_iters = 0
        #        elif not self.transcript_buffer.complete():
        #        #    logger.debug("NOT COMPLETE:",to_flush(self.transcript_buffer.complete()))
        #            self.silence_iters += 1
        #            if self.silence_iters >= 3:
        #                n = self.last_chunked_at
        # self.chunk_completed_sentence()
        # if n == self.last_chunked_at:
        #                self.chunk_at(self.last_chunked_at+self.chunk)
        #                logger.debug(f"\tCHUNK: 3-times silence! chunk_at {n}+{self.chunk}")
        # self.silence_iters = 0

        # if the audio buffer is longer than 30s, trim it...
        if len(self.audio_buffer) / self.SAMPLING_RATE >= 6:
            # ...on the last completed segment (labeled by Whisper)
            self.chunk_completed_segment(res)

            # alternative: on any word
            # l = self.buffer_time_offset + len(self.audio_buffer)/self.SAMPLING_RATE - 10
            # let's find commited word that is less
            # k = len(self.commited)-1
            # while k>0 and self.commited[k][1] > l:
            #    k -= 1
            # t = self.commited[k][1]
            logger.debug(f"chunking because of len")
            # self.chunk_at(t)

        logger.info(
            f"len of buffer now: {len(self.audio_buffer)/self.SAMPLING_RATE:2.2f}"
        )
        self.processing = False

        return self.to_flush(o)

    def chunk_completed_sentence(self):
        if self.commited == []:
            return
        # logger.debug(f"{self.commited}")
        sents = self.words_to_sentences(self.commited)
        for s in sents:
            logger.debug(f"\tSent: {s}")
        if len(sents) < 2:
            return
        while len(sents) >= 2:
            sents.pop(0)
        # we will continue with audio processing at this timestamp
        chunk_at = sents[-1][1]

        logger.debug(f"--- sentence chunked at {chunk_at:2.2f}")
        self.chunk_at(chunk_at)

    def chunk_completed_segment(self, res):
        if self.commited == []:
            return

        ends = self.segments_end_ts(res)

        t = self.commited[-1][1]

        if len(ends) >= 1:

            e = ends[-1] + self.buffer_time_offset
            while len(ends) > 2 and e > t:
                ends.pop(-1)
                e = ends[-1] + self.buffer_time_offset
            if e <= t:
                logger.debug(f"--- segment chunked at {e:2.2f}")
                self.chunk_at(e)
            else:
                logger.debug(f"--- last segment not within commited area")
        else:
            logger.debug(f"--- not enough segments to chunk")

    def segments_end_ts(self, res):
        return [s.end for s in res]

    def chunk_at(self, time):
        """trims the hypothesis and audio buffer at "time" """
        self.transcript_buffer.pop_commited(time)
        cut_seconds = time - self.buffer_time_offset
        self.audio_buffer = self.audio_buffer[int(cut_seconds) * self.SAMPLING_RATE :]
        self.buffer_time_offset = time
        self.last_chunked_at = time

    def words_to_sentences(self, words):
        """Uses self.tokenizer for sentence segmentation of words.
        Returns: [(beg,end,"sentence 1"),...]
        """

        cwords = [w for w in words]
        t = " ".join(o[2] for o in cwords)
        s = self.tokenizer.split(t)
        out = []
        while s:
            beg = None
            end = None
            sent = s.pop(0).strip()
            fsent = sent
            while cwords:
                b, e, w = cwords.pop(0)
                if beg is None and sent.startswith(w):
                    beg = b
                elif end is None and sent == w:
                    end = e
                    out.append((beg, end, fsent))
                    break
                sent = sent[len(w) :].strip()
        return out

    def finish(self):
        """Flush the incomplete text when the whole processing ends.
        Returns: the same format as self.process_iter()
        """
        o = self.transcript_buffer.complete()
        f = self.to_flush(o)
        logger.debug(f"last, noncommited: {f}")
        return f

    def to_flush(
        self,
        sents,
        sep=None,
        offset=0,
    ):
        # concatenates the timestamped words or sentences into one sequence that is flushed in one line
        # sents: [(beg1, end1, "sentence1"), ...] or [] if empty
        # return: (beg1,end-of-last-sentence,"concatenation of sentences") or (None, None, "") if empty
        if sep is None:
            sep = ""
        t = sep.join(s[2] for s in sents)
        if len(sents) == 0:
            b = None
            e = None
        else:
            b = offset + sents[0][0]
            e = offset + sents[-1][1]
        return (b, e, t)


WHISPER_LANG_CODES = "af,am,ar,as,az,ba,be,bg,bn,bo,br,bs,ca,cs,cy,da,de,el,en,es,et,eu,fa,fi,fo,fr,gl,gu,ha,haw,he,hi,hr,ht,hu,hy,id,is,it,ja,jw,ka,kk,km,kn,ko,la,lb,ln,lo,lt,lv,mg,mi,mk,ml,mn,mr,ms,mt,my,ne,nl,nn,no,oc,pa,pl,ps,pt,ro,ru,sa,sd,si,sk,sl,sn,so,sq,sr,su,sv,sw,ta,te,tg,th,tk,tl,tr,tt,uk,ur,uz,vi,yi,yo,zh".split(
    ","
)


def create_tokenizer(lan):
    """returns an object that has split function that works like the one of MosesTokenizer"""

    assert (
        lan in WHISPER_LANG_CODES
    ), "language must be Whisper's supported lang code: " + " ".join(WHISPER_LANG_CODES)


    # supported by fast-mosestokenizer
    if (
        lan
        in "as bn ca cs de el en es et fi fr ga gu hi hu is it kn lt lv ml mni mr nl or pa pl pt ro ru sk sl sv ta te yue zh".split()
    ):
        from mosestokenizer import MosesTokenizer

        return MosesTokenizer('en')

    # the following languages are in Whisper, but not in wtpsplit:
    if (
        lan
        in "as ba bo br bs fo haw hr ht jw lb ln lo mi nn oc sa sd sn so su sw tk tl tt".split()
    ):
        logger.debug(
            f"{lan} code is not supported by wtpsplit. Going to use None lang_code option."
        )
        lan = None

    from wtpsplit import WtP

    # downloads the model from huggingface on the first use
    wtp = WtP("wtp-canine-s-12l-no-adapters")

    class WtPtok:
        def split(self, sent):
            return wtp.split(sent, lang_code=lan)

    return WtPtok()
