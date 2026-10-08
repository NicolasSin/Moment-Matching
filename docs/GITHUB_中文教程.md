# 第一次把这个实验分享给教授

你已经有 GitHub 账号。这次创建 **Private（私有）仓库**，再邀请教授。

## 本次最快的方法：直接用网页

这个整理包的文件数量和单文件大小都在网页上传限制内，可以先不安装任何软件：

1. 解压最终的 `lorenz63-moment-matching.zip`，找到直接包含 `README.md` 的那一层。
2. 登录 GitHub，打开 [新建仓库](https://github.com/new)。名称填 `lorenz63-moment-matching`，选择 **Private**，不另外初始化 README，然后点击 **Create repository**。
3. 在空仓库页面点击 **uploading an existing file**。拖入整理包里面的文件和文件夹，使 `README.md` 位于仓库最外层。不要只上传 ZIP 文件。
4. 等待上传完成，在说明中填 `Initial K1500 GH4000 experiment`，点击 **Commit changes**。这一步把文件和第一份版本记录一起保存到 GitHub。
5. 在仓库的 **Settings → Collaborators → Add people** 邀请教授的 GitHub 账号。教授接受后，把仓库网址发给教授。

若文件没有全部出现，先核对网页中的目录再分享。私有仓库只有被授权的人能访问；仅发链接还不够。

下面是适合以后经常修改代码时使用的 GitHub Desktop 方法。选一种上传方法即可。

## 1. 安装并登录 GitHub Desktop

从 [GitHub Desktop 官网](https://desktop.github.com/) 下载，安装后选择 **Sign in to GitHub.com**，在浏览器登录自己的账号。

## 2. 创建一个新的本地仓库

在 Desktop 点击 **File → New repository**：

- Name：`lorenz63-moment-matching`
- Description：`Scalar Lorenz-63 moment matching: K1500, Adam1000 + GH4000`
- Local path：选择一个方便的位置，例如桌面上的新文件夹。
- 不需要自动生成 README；整理包中已经有了。
- License 暂时不选。本次是私下分享研究材料，没有替你决定开源授权。

创建后，打开这个仓库文件夹，把整理包 `lorenz63-moment-matching` **里面的文件和子文件夹**复制进去。最外层应直接看到 `README.md`、`code`、`scripts`、`data`、`models`、`results`，不要再套一层同名文件夹。

仅复制整理包内部内容；旁边的 `_maintenance` 是本次本地整理用文件，不属于分享包。

## 3. 保存第一版，再上传

回到 Desktop，在 **Changes** 查看文件。左下角 **Summary** 输入：

`Initial K1500 GH4000 experiment`

点击 **Commit to main**（实际按钮名称随本地分支名变化），再点击上方 **Publish repository**。

**保持勾选 `Keep this code private`**，确认名称后发布。

记住三个词即可：

| 操作 | 含义 |
| --- | --- |
| Repository | 项目的文件夹及版本记录，简称仓库 |
| Commit | 在你的电脑上保存一个有说明的版本 |
| Publish / Push | 首次上传 / 将后续版本上传 GitHub |

**只 Commit 不会把文件传到网页上。**

## 4. 邀请教授

在 Desktop 选择 **Repository → View on GitHub** 打开网页。确认仓库名称旁显示 **Private**。

进入 **Settings → Collaborators → Add people**，输入教授的 GitHub 用户名，发出邀请。教授接受后才能查看；然后你可以把仓库网址发给教授。

个人私有仓库的 collaborator 有读写权限。如果教授看到 404，检查是否登录了受邀账号并接受邀请。

## 5. 以后更新

修改本地仓库的文件后：

1. 回 Desktop 查看 Changes。
2. 写一句本次改了什么。
3. **Commit to main**。
4. **Push origin**。

若教授也会改文件，开始工作前先 **Fetch origin**，有更新时 **Pull origin**。

## 教授怎么使用

- 网页首页直接阅读 README 和结果图。
- **Code → Download ZIP** 下载全部文件。
- 阅读 `docs/TRAINING.md` 和 `docs/EVALUATION.md` 运行实验。
- 保存的模型及结果已经在仓库内，查看结果不需要重新训练。

## 不安装 Desktop 的备用方法

登录 [GitHub](https://github.com/)，右上角 **+ → New repository**，选择 **Private**。创建后点 **Add file → Upload files**（空仓库可能显示 `uploading an existing file`），拖入解压后的内容，再提交。

网页每次最多100个文件，每个文件最多25 MiB。普通 Git 会拒绝超过100 MiB的单个文件。此整理包会在交付时检查单文件大小；以后产生的新大文件放在已忽略的 `outputs/`，不要把虚拟环境一起上传。

## 官方说明

- [Desktop 首次创建并发布仓库](https://docs.github.com/en/desktop/overview/creating-your-first-repository-using-github-desktop)
- [邀请教授进入个人私有仓库](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/repository-access-and-collaboration/inviting-collaborators-to-a-personal-repository)
- [个人仓库协作者权限](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/repository-access-and-collaboration/permission-levels-for-a-personal-account-repository)
- [网页上传文件](https://docs.github.com/en/repositories/working-with-files/managing-files/adding-a-file-to-a-repository)
- [文件大小限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)
