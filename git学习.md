# git学习

## 常用命令

> git log --oneline --all -n4 --graph

- --oneline 一行显示
- --all 所有分支
- -n4 最近四条
- --graph 图形化

> git branch

- -av 查看所有分支
- -d 或者 -D 删除分支

> gitk --all

- 图形化界面看分支

> git checkout 某个分支

- 切换分支
- 加上参数 -b 创建新分支并切换
- （危险命令） -- 文件名  将工作区的这个文件变成暂存区的

> git commit --amend

- 最近一次提交的message做变更

> git rebase -i

- 交互式，可以修改历史提交的message
- 可以合并历史的提交
- 等等

> git diff 

- 查看修改的地方
- --cached
- -- 文件名  查看具体文件修改的地方

> git reset HEAD

- 恢复暂存区，和head一致
- -- 文件名  将暂存区的这个文件撤销

## 小tips

- init一个仓库，创建一个doc文件夹，再在里面添加一个文件readme，内容是hello world。提交之后，.git/objects文件夹下有四个文件。
  - 一个commit，里面有一个tree
  - tree里面有一个tree，内容是doc
  - doc那个tree里面有一个blob，里面内容是readme
  - blob里面内容是hello world