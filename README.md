# Ethan-a2.github.io

Ethan-a2 的个人主页仓库，用作主域名总入口。页面采用《桃花源记》灵感的亮色田园风格，用于集中跳转到博客、作品集、GitHub 和其他常用链接。

## 文件结构

- `index.html`：主页入口。
- `404.html`：GitHub Pages 的迷路页面，沿用“不复得路”的桃源风格。
- `site.config.js`：站点配置，导航、快捷链接和作品集都从这里维护。
- `styles.css`：页面视觉样式。
- `script.js`：把配置渲染到页面中。
- `.nojekyll`：关闭 Jekyll 处理，按静态文件直接发布。

## 扩展导航

编辑 `site.config.js` 里的 `navigation`：

```js
navigation: [
  { label: "博客", href: "/blog/" },
  { label: "桃花源记", href: "#taohuayuan" },
  { label: "作品集", href: "#portfolio" },
  { label: "GitHub", href: "https://github.com/Ethan-a2", external: true },
]
```

外链建议加 `external: true`，会自动使用新窗口打开。

## 扩展作品集

编辑 `site.config.js` 里的 `portfolio`：

```js
portfolio: [
  {
    name: "项目名称",
    type: "Project",
    href: "https://example.com",
    description: "一句话介绍项目亮点。",
  },
]
```

## 发布

把仓库推送到 `Ethan-a2/Ethan-a2.github.io` 后，在 GitHub Pages 中选择从 `main` 分支发布即可。默认访问地址为：

```text
https://ethan-a2.github.io/
```
