FROM waggle/plugin-base:1.1.1-ml-cuda10.2-l4t

COPY requirements.txt /app/
RUN pip3 install --no-cache-dir -r /app/requirements.txt

COPY unet /app/unet
COPY app.py unet_module.py image.jpg /app/

ADD https://web.lcrc.anl.gov/public/waggle/models/Unet_epoch228a.pth /app/wagglecloud_unet_300.pth

WORKDIR /app
ENTRYPOINT ["python3", "-u", "/app/app.py"]
