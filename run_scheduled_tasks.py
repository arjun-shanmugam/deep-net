import schedule
import time
from utils.send_email import send_email
from utils.run_git_commands import git_push_updated_db
from deep_net.database import Database
import argparse

def daily_tasks(db):
    #Fetch updated data
    db.update_games_table()
    db.update_boxscores_table()
    send_email(["1"]) #TODO: Make preds, and email them
    git_push_updated_db()
    print("Pushed to remote")

def run(db):
    # time.sleep(30) #sleep 30 secs (testing purposes)
    # schedule.every(2).minutes.do(daily_tasks, db)
    schedule.every().day.at("09:00").do(daily_tasks, db) #9 AM
    while True:
        schedule.run_pending()
        time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_tasks_now', dest='run_tasks_now', action='store_true')
    args = parser.parse_args()
    db = Database("data/nbastats.db")
    if args.run_tasks_now:
        daily_tasks(db)
    run(db)