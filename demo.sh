
python whisper_online_server.py --language zh --min-chunk-size 1 command > output.txt 2> error.txt
python whisper_online_server.py --language zh --min-chunk-size 1 > out.txt

python whiser_online.py

ffmpeg -i rtmp://your-server-ip/myapp/stream_key -acodec pcm_s16le -f s16le tcp://localhost:43007

ffmpeg -i sample_video.wav -ac 1 -ar 16000 -f wav tcp://localhost:43007

ffmpeg -i sample_video.wav -ac 1 -ar 16000 -f wav - | nc -N localhost 43007

ffmpeg -f alsa -i default -acodec pcm_s16le -ar 44100 -ac 2 -f wav - | nc -N 43007
HOSTNAME_OR_IP PORT


python whisper_online.py  sample_video.wav --language zh --min-chunk-size 1 > out.txt

ffmpeg -i sample_video.wav -ac 1 -ar 16000 -tune zerolatency -muxdelay 0 -af "afftdn=nf=-20, highpass=f=200, lowpass=f=3000" -vn -sn -dn -f wav -ar 16000 - 2>/dev/null | node app.js 
$feed_time

ffmpeg -i $RTMP_INPUT  -ac 1 


docker run --gpus all -p 43008:43007 whisper-streaming-server:amazonlinux 
docker run --gpus all -p 43008:43007 -d 185958750037.dkr.ecr.ap-northeast-1.amazonaws.com/whisper-streaming-server:latest python3 whisper_online_server.py --language en --min-chunk-size 1
