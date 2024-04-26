export FILE_NAME=../../aps208.mp3


export ASR_LANG=zh
export ASR_LANG=en
export FILE_NAME=../../sample_video.wav
ffmpeg -i $FILE_NAME -ac 1 -ar 16000 -tune zerolatency -muxdelay 0 -af "afftdn=nf=-20, highpass=f=200, lowpass=f=3000" -vn -sn -dn -f wav -ar 16000 - 2>/dev/null | node app.js 