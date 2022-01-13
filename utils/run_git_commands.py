import subprocess as sp
import datetime

#Runs all git commands to push the updated DB file to remote repo
def git_push_updated_db():
    sp.run(["git", "fetch"])
    sp.run(["git", "pull"])
    sp.run(["git", "add", "-A"])
    sp.run(["git", "commit", "-m", "Automated DB Update Commit - ".format(datetime.date.today())])
    sp.run(["git", "push"])