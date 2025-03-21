#!/bin/bash

# 生成 requirements.txt
echo "Generating requirements.txt..."
pip freeze > requirements.txt

# 获取当前时间
current_time=$(date "+%Y-%m-%d %H:%M:%S")

# 获取变更的文件信息
echo "Checking changed files..."
# 获取新增的文件
new_files=$(git status --porcelain | grep "^??" | cut -c4- | tr '\n' ',' | sed 's/,$/\n/')
# 获取修改的文件
modified_files=$(git status --porcelain | grep "^ M\|^M" | cut -c4- | tr '\n' ',' | sed 's/,$/\n/')

# 构建提交信息
commit_message="Auto commit at ${current_time}\n\n"

if [ ! -z "$new_files" ]; then
    commit_message+="New files:\n${new_files}\n\n"
fi

if [ ! -z "$modified_files" ]; then
    commit_message+="Modified files:\n${modified_files}\n"
fi

# Git 操作
echo "Adding files to git..."
git add .

echo "Committing changes..."
git commit -m "$(echo -e ${commit_message})"

echo "Pushing to remote repository..."
git push

echo "Done! Code has been committed and pushed successfully." 