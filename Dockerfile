FROM waggle/plugin-base:1.1.1-ml-cuda10.2-l4t

COPY requirements.txt /app/
RUN apt-get install python3.8-tk
RUN pip3 install --no-cache-dir -r /app/requirements.txt

COPY app.py module.py /app/

ADD https://web.lcrc.anl.gov/public/waggle/models/cloudcover_best_model_unet_epoch81.pth /app/new_cloudcover_unet.pth

WORKDIR /app
ENTRYPOINT ["python3", "-u", "/app/app.py"]
