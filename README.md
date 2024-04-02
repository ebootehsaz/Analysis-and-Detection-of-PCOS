# Analysis-and-Detection-of-PCOS

Our goal is to develop a predictive model for the early detection of Polycystic ovary syndrome (PCOS) and infertility-related issues. PCOS is a complex hormonal disorder, and early diagnosis is crucial for effective management.

We will analyze the dataset to identify patterns and trends. We will develop a model that can predict the likelihood of PCOS and infertility-related issues. We will evaluate the model's performance using various metrics. We will present the results in a clear and concise manner.

## Table of Contents

- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Navigation](#navigation)
- [Clone the Repository](#clone-the-repository)
- [Branching](#branching)
- [Creating a Pull Request](#creating-a-pull-request)
- [Goals](#goals)
- [Other Information](#other-information)
- [License](#license)
- [Authors](#authors)

## Presentation 1 (Progress Report)

- Explore the PCOS dataset to understand its structure and features.
- Identify missing values, outliers, and patterns in the data.
- Select relevant physical and clinical parameters for PCOS and infertility detection.
    - Assessment of Ovarian Reserve: AMH (Anti-Müllerian Hormone) levels give insight into the ovarian reserve, which is crucial for fertility.
    - Women with PCOS often have higher than normal AMH levels. This is because AMH is produced by the granulosa cells of ovarian follicles, and women with PCOS tend to have a higher number of small follicles in their ovaries.
- Perform any necessary preprocessing steps, such as handling missing values or encoding categorical variables.
- Choose appropriate machine learning models for binary classification (e.g., Logistic Regression)
- Formulate hypotheses related to PCOS and infertility based on the dataset.
- Create visualizations to support your findings.

- Generate tables/graphs to show the distribution of the target variable and key features.


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Prerequisites

Use a virtual environment to install the required packages. 

I have created a bash script to do this for you. You can run the following command to make and activate the virtual environment, as well as pull recent changes:

```
. ./start.sh
or 
source ./start.sh
```

Whenever you add a new library to the project, you can update the requirements.txt file using the following command:

```
pip freeze > requirements.txt
```

## Navigation

The project is divided into the following directories:

- `data`: Contains the dataset used in the project, after being cleaned.
- `src`: Contains the source code used in the project.
- `Kaggle_Data`: Contains the dataset used from Kaggle.
- `tests`: Contains the test cases used in the project. tba.
- `notebooks`: Contains the Jupyter notebooks used in the project. tba.


## Kanban Board

We will be using a Kanban board to track the progress of the project. You can access the Kanban board using the following link:
https://github.com/ebootehsaz/Analysis-and-Detection-of-PCOS/projects?query=is%3Aopen

Think of the Kanban board as a to-do list. You can create issues, assign tasks, and track progress on the Kanban board. You can move tasks from one column to another as you complete them. You can also add labels, assignees, and due dates to tasks on the Kanban board. You can use the Kanban board to organize your work, prioritize tasks, and track progress on the project.


## Clone the Repository

You can clone the repository using the following command:

```
git clone https://github.com/ebootehsaz/Analysis-and-Detection-of-PCOS
```

You can make commits to the repository using the following commands:

```
git add .
git commit -m "Your message here"
git push
```

The remote repository is updated with the latest changes.
Write a detailed commit message to help others understand the changes you have made.

## Making use of GitHub

GitHub is a platform that allows you to collaborate with others on projects. You can use GitHub to share your code with others, track changes to your code, and collaborate with others on projects. You can also use GitHub to create issues, assign tasks, and track progress on projects. You can use GitHub to create branches, merge changes, and create pull requests. You can use GitHub to review others' code, provide feedback, and suggest changes. 

Naturally, others will be saving their changes to the repository as well. You can pull the latest changes from the repository using the following command:

```
git pull
```

This will update your local repository with the latest changes from the remote repository. Make sure to resolve any conflicts that may arise when pulling changes from the remote repository. You can resolve conflicts by editing the files that have conflicts and then committing the changes.

RUN THIS EVERY TIME YOU START WORKING ON THE PROJECT!

I created a bash script to do this for you as well, so just run

```
. ./start.sh
or 
source ./start.sh
```


## Branching

I recommend creating a new branch for each feature or bug fix you work on. This will help you keep your changes organized and make it easier to merge your changes into the main branch. It will also help you avoid conflicts with other developers who are working on the same project. Furthermore, it will give others the opportunity to review your changes before they are merged into the main branch, and it will give you the opportunity to review others' changes before they are merged into the main branch. Good experience!

You can create a new branch to work on a new feature or bug fix. To create a new branch, follow these steps:

```
git checkout -b branch_name
```

You can switch between branches using the following command:

```
git checkout branch_name
```

You can commit and push your changes to the branch using the following commands:

```
git add .
git commit -m "Your message here"
git push origin branch_name
```

You can delete a branch using the following command:

```
git branch -d branch_name
```

You can merge a branch into the main branch by creating a pull request.

## Creating a Pull Request

You can create a pull request to merge your changes into the main branch. To create a pull request, follow these steps:

1. Go to the repository on GitHub.
2. Click on the "Pull Requests" tab.
3. Click on the "New Pull Request" button.
4. Select the branch you want to merge into the main branch.
5. Write a detailed description of the changes you have made.
6. Click on the "Create Pull Request" button.

Your pull request will be reviewed by others before it is merged into the main branch. Make sure to write a detailed description of the changes you have made to help others understand your changes.

## Goals

- Develop a predictive model for the early detection of PCOS and infertility-related issues.
- Analyze the dataset to identify patterns and trends.
- Develop a model that can predict the likelihood of PCOS and infertility-related issues.
- Evaluate the model's performance using various metrics.
- Present the results in a clear and concise manner.


## Other Information

VSCode is the recommended IDE for this project. You can download it from the following link: https://code.visualstudio.com/

Extension recommendations:

- Python
- Pylance
- Jupyter
- Jupyter slideshow
- Rainbow CSV
- Markdownlint
- JSON
- Copilot
- Excel Viewer

You can install the extensions by searching for them in the Extensions tab in VSCode.

Some useful tips and tricks:

- Use the command palette (Ctrl+Shift+P) to quickly access commands.
- Use the integrated terminal to run commands.
- Use copilot to generate code snippets.
- Ask questions if you are stuck.
- Use a virtual environment.
- Ask ChatGPT questions, but don't rely on it too much.

You can move through the project in the terminal and run commands like so:

```
cd src
python main.py
```

You can also run the project in the terminal using the following command:

```
python src/main.py
```

You can verify that you are using a virtual environment by runing the following command:

```
which python
```

You should see the path to the virtual environment.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Authors

Ethan Bootehsaz (ebootehsaz)
member2
member3
member4