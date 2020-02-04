# How to start up with server

1. Create vitrual env with the following command:

python3 -m venv venv/

2. Then activate it:

source venv/bin/activate

3. Load important packages:

pip install requests 

pip install numpy

pip install keras==2.2.5 

pip install tensorflow==1.14 

pip install time

pip install pyOpenSSL

pip install flask

4. It is time to load trained model with

python load_model1.py

5. Do one test

python test.py

6. Run server

python server.py

7. Find %dunno what is it% (for me it is 'http://127.0.0.1:5000/'). And remember last nuber it as %local%

8. Install, LogIn/Register in this service https://ngrok.com. Make intallation in current folder 

9. Run in terminal 
ngrok http %local%

10. From string starting with 'Froward' get an URL to use

IMPORTANT You have to use tensorflow version = 1.15 

If you have an error "illegal instruction" use version = 1.15


# BJokeNN_bf
Web site generates dumb jokes

Here you can find a .ipynd which is to be launched in collab following my instructions:

1. Go to google drive and create "BAneks" foder in root
2. Upload ipython notebook to the folder
3. Run all
4. Report about problems

# Parser
This notebook helps parse data, loaded from vk.brakov.net

# test
This is an example of using text generator

It worth to mkdir 'models' and store weights there
