Exercises for Econometrics B (spring 2022).

This folder has the necessary files to do the exercises. At the end of the day, I will upload an ex_post folder.

To get this repository, follow the instructions at https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository. The short story:

- Click "Code" and copy the "HTTPS" link.
- On your computer open you terminal and navigate to a local folder that you want to place the repo at.
- Use the command git clone link, where link is the copied link.

The changes that you make in these files should be safe, as long as you do not initate another git pull or git fetch. So to make git download the ex_post files for you, you need to first "shield" your local changes. This can be done in the following way:

- Open your terminal and navigate to the folder that you cloned the repo at.
- First, run the command git stash to "stash" away your local changes.
- Then run git pull to pull the repository again.
- Finally, run git stash pop to bring back your local changes.
- You should now have the ex_post files downloaded, and your local changes are unaffected.
