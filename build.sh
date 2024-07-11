# CI

aws ecr get-login-password --region ap-northeast-1 --profile ray-cluster | docker login --username AWS --password-stdin 464616699298.dkr.ecr.ap-northeast-1.amazonaws.com
docker build -t streaming-asr . 
docker tag streaming-asr 464616699298.dkr.ecr.ap-northeast-1.amazonaws.com/streaming-asr
docker push 464616699298.dkr.ecr.ap-northeast-1.amazonaws.com/streaming-asr  


docker tag streaming-asr public.ecr.aws/darrenlin/whisper-streaming-v2:latest
docker push public.ecr.aws/darrenlin/whisper-streaming-v2:latest

# CD 
kubectl delete 
kubeclt apply 

