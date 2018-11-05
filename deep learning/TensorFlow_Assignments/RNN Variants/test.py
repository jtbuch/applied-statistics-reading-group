


import os




MODEL_DIR = os.path.join(os.getcwd(), "model_files")




logdir =os.path.join(os.getcwd(), MODEL_DIR) 
os.system("Tensorboard --logdir="+logdir)

print "\nenter into web broser:"

print "http://localhost:xxxxx"

print "(where \"xxxxx\" is the port number printed above)\n"