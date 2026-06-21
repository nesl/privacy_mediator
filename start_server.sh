source venv/bin/activate
nohup zrok2 share public 5000 --headless &> zrok_output.log &
nohup python survey/server.py   --k 25   --pipeline-output-dir runs/context_pipeline_generation
 &> output.log &
