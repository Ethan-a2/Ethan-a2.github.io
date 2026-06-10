# Ethan-a2.github.io 本地使用说明

这个仓库是一个纯静态 GitHub Pages 个人主页，用作主域名总入口。页面采用 404 / terminal 风格，可以集中放博客、作品集、GitHub、在线简历和其他常用链接。

## 1. 你需要知道的文件

日常使用时，主要关注这些文件：

| 文件 | 作用 |
| --- | --- |
| `index.html` | 首页入口，不常改 |
| `404.html` | GitHub Pages 的 404 页面，不常改 |
| `site.config.js` | 站点配置，最常改 |
| `styles.css` | 页面样式，需要改视觉时再改 |
| `script.js` | 把配置渲染到页面，不常改 |
| `favicon.svg` | 浏览器标签页图标 |
| `.nojekyll` | 关闭 Jekyll，保持静态文件原样发布 |

最重要的是：**新增导航、修改博客地址、添加作品集，优先改 `site.config.js`。**

## 2. 本地预览

在仓库根目录执行：

```bash
python3 -m http.server 4173 --bind 127.0.0.1
```

然后在浏览器打开：

```text
http://127.0.0.1:4173/
```

预览 404 页面：

```text
http://127.0.0.1:4173/404.html
```

停止本地服务：

```text
Ctrl + C
```

如果使用 VS Code，也可以安装 `Live Server` 插件，然后右键 `index.html` 选择本地预览。

## 3. 修改个人介绍

打开 `site.config.js`，找到：

```js
owner: {
  name: "Ethan-a2",
  handle: "@Ethan-a2",
  tagline: "个人主页 · 博客入口 · 作品集导航",
  intro:
    "这里是主域名总入口。把博客、项目、作品集和常用链接集中到一个 404 灵感的个人导航页。",
},
```

可以改成：

```js
owner: {
  name: "Ethan",
  handle: "@Ethan-a2",
  tagline: "开发者 · 技术博客 · 作品集",
  intro:
    "这里是我的个人主页，用于收录博客、项目、作品集和常用链接。",
},
```

首页介绍文字会自动更新。

## 4. 修改顶部导航

顶部导航由 `navigation` 控制：

```js
navigation: [
  { label: "博客", href: "/blog/" },
  { label: "作品集", href: "#portfolio" },
  { label: "链接", href: "#links" },
  { label: "GitHub", href: "https://github.com/Ethan-a2", external: true },
],
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `label` | 页面上显示的名称 |
| `href` | 点击后跳转的地址 |
| `external` | 是否外链，新窗口打开 |

新增一个“关于我”：

```js
{ label: "关于我", href: "/about/" },
```

新增一个外部链接：

```js
{ label: "Bilibili", href: "https://space.bilibili.com/你的ID", external: true },
```

## 5. 修改首页快捷入口

首页中间的卡片由 `quickLinks` 控制：

```js
quickLinks: [
  {
    label: "Blog",
    title: "博客 / Notes",
    href: "/blog/",
    description: "文章、学习记录和长期笔记的入口。",
    accent: "cyan",
  },
],
```

字段说明：

| 字段 | 说明 |
| --- | --- |
| `label` | 卡片左上角小标签 |
| `title` | 卡片标题 |
| `href` | 点击地址 |
| `description` | 卡片说明 |
| `accent` | 强调色 |
| `external` | 是否外链 |

可用强调色：

```text
cyan, violet, green, orange
```

示例：添加在线简历入口。

```js
{
  label: "Resume",
  title: "在线简历",
  href: "/resume/",
  description: "我的经历、技能栈和联系方式。",
  accent: "orange",
}
```

## 6. 修改作品集

作品集区域由 `portfolio` 控制：

```js
portfolio: [
  {
    name: "作品集占位 A",
    type: "Project",
    href: "#",
    description: "替换为你的项目地址、在线 Demo 或案例页面。",
  },
],
```

改成真实项目：

```js
portfolio: [
  {
    name: "个人博客系统",
    type: "Blog",
    href: "https://ethan-a2.github.io/blog/",
    description: "用于记录技术文章、学习笔记和项目复盘。",
    external: true,
  },
  {
    name: "高仿 404 页面合集",
    type: "Design",
    href: "/404.html",
    description: "基于 terminal / glitch 风格的错误页视觉实验。",
  },
  {
    name: "工具项目示例",
    type: "Tool",
    href: "https://github.com/Ethan-a2/project-name",
    description: "一个用于展示工具类项目的作品集入口。",
    external: true,
  },
],
```

## 7. 博客地址怎么设置

当前默认博客地址是：

```text
/blog/
```

最终会访问：

```text
https://ethan-a2.github.io/blog/
```

常见用法：

### 博客放在同一个仓库

创建：

```text
blog/index.html
```

然后保持地址：

```js
{ label: "博客", href: "/blog/" }
```

### 博客是另一个 GitHub Pages 项目

例如博客地址是：

```text
https://ethan-a2.github.io/my-blog/
```

就改成：

```js
{ label: "博客", href: "/my-blog/" }
```

### 博客是外部域名

例如：

```text
https://blog.example.com
```

就改成：

```js
{ label: "博客", href: "https://blog.example.com", external: true }
```

## 8. 发布到 GitHub Pages

确认仓库名是：

```text
Ethan-a2.github.io
```

提交代码：

```bash
git add .
git commit -m "Init personal homepage"
git push origin main
```

打开 GitHub 仓库：

```text
Settings → Pages
```

如果没有自动发布，选择：

```text
Source: Deploy from a branch
Branch: main
Folder: /root
```

发布地址：

```text
https://ethan-a2.github.io/
```

首次发布可能需要等待 1 到 5 分钟。

## 9. 绑定自定义域名

如果以后有自己的域名，例如：

```text
ethan.example.com
```

在仓库根目录新增 `CNAME` 文件，内容只写域名：

```text
ethan.example.com
```

然后在域名 DNS 里添加 CNAME 记录：

```text
ethan.example.com → Ethan-a2.github.io
```

最后到 GitHub Pages 设置里填写 Custom domain。

## 10. 日常维护流程

每次更新主页可以按这个流程：

```bash
# 1. 修改 site.config.js

# 2. 本地预览
python3 -m http.server 4173 --bind 127.0.0.1

# 3. 提交
git add .
git commit -m "Update homepage links"

# 4. 推送发布
git push origin main
```

## 11. 常见问题

### 修改后线上没变化

可能是 GitHub Pages 还没部署完成，等待几分钟后刷新。也可以在仓库的 `Actions` 或 `Settings → Pages` 中查看部署状态。

### 点击博客后 404

说明 `/blog/` 这个路径下暂时没有页面。需要创建 `blog/index.html`，或者把 `site.config.js` 中的博客链接改成真实博客地址。

### 外部链接没有新窗口打开

外链配置里加上：

```js
external: true
```

### 作品集还是占位内容

打开 `site.config.js`，把 `portfolio` 里的 `作品集占位 A/B/C` 替换成真实项目。

### 想换页面风格

修改 `styles.css`。如果只是换文字和链接，不需要改样式文件。

