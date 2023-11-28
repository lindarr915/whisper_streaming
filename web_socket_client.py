import asyncio
import websockets

async def send_audio():
    uri = "ws://localhost:8000/audio"  # Replace with your server's address
    chunk_size = 1024  # Adjust the chunk size as needed

    with open("../sample_video.wav", "rb") as audio_file:
        while True:
            audio_chunk = audio_file.read(chunk_size)
            if not audio_chunk:
                break
            async with websockets.connect(uri) as websocket:
                await websocket.send(audio_chunk)
            await asyncio.sleep(0.1)  # Adjust the delay between chunks

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(send_audio())
