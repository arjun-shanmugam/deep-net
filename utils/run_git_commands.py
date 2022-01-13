import subprocess as sp
import datetime

#Runs all git commands to push the updated DB file to remote repo
def git_push_updated_db():
    sp.run(["git", "fetch"])
    sp.run(["git", "pull"])
    sp.run(["git", "add", "data/nbastats.db"])
    now = datetime.now()
    current_date_and_time = now.strftime("%m/%d/%Y, %H:%M:%S")
    sp.run(["git", "commit", "-m", "Automated DB Update Commit - {}".format(current_date_and_time)])
    sp.run(["git", "push"])