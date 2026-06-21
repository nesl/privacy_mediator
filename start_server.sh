source ../venv/bin/activate
nohup zrok share public 9890 --headless &> zrok_output.log &
nohup python deepseek_server.py &> output.log &
