import schedule
import time
from utils.send_email import send_email
from database import Database
from git import Repo
from utils.run_git_commands import git_push_updated_db
 
def daily_tasks(db, git_repo):
    #Fetch updated data
    db.update_games_table()
    db.update_boxscores_table()
    send_email(["1"]) #TODO: Make preds, and email them
    git_push_updated_db()

  
schedule.every(5).minutes.do(daily_tasks) #9 AM

def run():
    repo = git.Repo('deep-net')
    db = Database("../data/nbastats.db")
    while True:
        schedule.run_pending(db, repo)
        time.sleep(1)

if __name__ == "__main__":
    run()