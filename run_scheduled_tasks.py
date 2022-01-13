import schedule
import time
from utils.send_email import send_email
from utils.run_git_commands import git_push_updated_db
from deep_net.database import Database
 
def daily_tasks(db):
    #Fetch updated data
    db.update_games_table()
    db.update_boxscores_table()
    send_email(["1"]) #TODO: Make preds, and email them
    git_push_updated_db()

def run():
    db = Database("data/nbastats.db")
    schedule.every(5).minutes.do(daily_tasks(db=db)) #9 AM
    time.sleep(120) #sleep 2 mins (testing purposes)
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    run()