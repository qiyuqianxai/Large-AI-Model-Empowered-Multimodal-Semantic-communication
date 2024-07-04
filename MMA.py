import json
import os
from CoDi.core.models.model_module_infer import model_module
from matplotlib import pyplot as plt
from PIL import Image
import torchaudio
from IPython.display import Audio
import cv2

# Image to Text using CoDi
def img2text(dataset):
    for img in os.listdir(dataset):
        if not img.endswith(".jpg"):
            continue
        img_path = os.path.join(dataset,img)
        dst_txt_path = img_path.replace(".jpg",".json")
        if os.path.exists(dst_txt_path):
            continue
        im = Image.open(img_path).resize((224, 224))
        text = inference_tester.inference(
            xtype=['text'],
            condition=[im],
            condition_types=['image'],
            n_samples=5,
            ddim_steps=10,
            scale=7.5, )
        text = text[0]
        with open(dst_txt_path,"w",encoding="utf-8") as f:
            f.write(json.dumps(text,indent=4,ensure_ascii=False))
    return text

# Audio to Text using CoDi
def audio2text(dataset):
    for audio in os.listdir(dataset):
        if not audio.endswith(".flac"):
            continue
        audio_path = os.path.join(dataset,audio)
        dst_txt_path = audio_path.replace(".flac",".json")
        if os.path.exists(dst_txt_path):
            continue

        audio_wavs, sr = torchaudio.load(audio_path)
        audio_wavs = torchaudio.functional.resample(waveform=audio_wavs, orig_freq=sr, new_freq=16000).mean(0)[
                     :int(16000 * 10.23)]
        # Audio(audio_wavs.squeeze(), rate=16000)
        text = inference_tester.inference(
            xtype=['text'],
            condition=[audio_wavs],
            condition_types=['audio'],
            n_samples=2,
            ddim_steps=10,
            scale=7.5)
        text = text[0]
        with open(dst_txt_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(text, indent=4, ensure_ascii=False))
    return text

# Video to Text using CoDi
def video2text(dataset):
    for video in os.listdir(dataset):
        if not video.endswith(".avi"):
            continue
        video_path = os.path.join(dataset,video)
        dst_txt_path = video_path.replace(".avi",".json")
        if os.path.exists(dst_txt_path):
            continue
        video_text = []
        cv = cv2.VideoCapture(video_path)
        if cv.isOpened():
            rval, frame = cv.read()
            i = 0
        else:
            rval = False
            print('open video error!!')
        while rval:
            rval, frame = cv.read()
            if (i % 10 == 0):
                try:
                    im = Image.fromarray(frame)
                except:
                    continue
            i += 1
            if i > 3:
                break
            # Audio(audio_wavs.squeeze(), rate=16000)
            text = inference_tester.inference(
                xtype=['text'],
                condition=[im],
                condition_types=['image'],
                n_samples=1,
                ddim_steps=10,
                scale=7.5)
            video_text += text[0]
        cv2.waitKey(1)
        cv.release()
        with open(dst_txt_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(video_text, indent=4, ensure_ascii=False))
    return video_text

# Text to image using CoDi
def text2img(prompts):
    for i,prompt in enumerate(prompts):
        images = inference_tester.inference(xtype = ['image'],
                        condition = [prompt],
                        condition_types = ['text'],
                        n_samples = 1,
                        image_size = 256,
                        ddim_steps = 50)
        plt.imshow(images[0][0])
        plt.axis('off')
        plt.savefig(f"text2img_res{i}.jpg")
        # plt.show()

# Text to video using CoDi
def text2video(prompts):
    for i,prompt in enumerate(prompts):
        outputs = inference_tester.inference(
            ['video'],
            condition=[prompt],
            condition_types=['text'],
            n_samples=1,
            image_size=256,
            ddim_steps=50,
            num_frames=8,
            scale=7.5)

        video = outputs[0][0]
        # Visual video as gif
        from PIL import Image
        frame_one = video[0]
        path = f"text2video_res{i}.gif"
        frame_one.save(path, format="GIF", append_images=video[1:],
                       save_all=True, duration=2000 / len(video), loop=0)

# Text to audio using CoDi
def text2audio(prompts):
    # Generate audio
    for i,prompt in enumerate(prompts):
        audio_wave = inference_tester.inference(
            xtype=['audio'],
            condition=[prompt],
            condition_types=['text'],
            scale=7.5,
            n_samples=1,
            ddim_steps=50)[0]
        torchaudio.save(f"text2audio_res{i}.wav", audio_wave, 16000)
    # # Play the audio
    # from IPython.display import Audio
    # Audio(audio_wave.squeeze(), rate=16000)

# model weights path
model_load_paths = ['CoDi_encoders.pth', 'CoDi_text_diffuser.pth', 'CoDi_audio_diffuser_m.pth',
                        'CoDi_video_diffuser_8frames.pth']
# load CoDi model
inference_tester = model_module(data_dir='./', pth=model_load_paths)
inference_tester = inference_tester
inference_tester = inference_tester.eval()

if __name__ == '__main__':
    img_dataset = r"E:\pengyubo\datasets\DeepJSCC_res"
    img2text(img_dataset)
    audio_dataset = r"E:\pengyubo\datasets\Fairseq_res"
    audio2text(audio_dataset)
    video_dataset = r"E:\pengyubo\datasets\MMA_test"
    video2text(video_dataset)


