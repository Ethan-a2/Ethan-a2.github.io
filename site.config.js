window.siteConfig = {
  owner: {
    name: "Ethan-a2",
    handle: "@Ethan-a2",
    tagline: "桃花源记 · 博客入口 · 作品集导航",
    intro:
      "以一卷清亮纸本作主页，在桃花、溪水与远山之间，收纳博客、作品集和常用链接。",
  },
  navigation: [
    { label: "桃源入口", href: "#links" },
    { label: "阡陌作品集", href: "#portfolio" },
    { label: "桃花源记", href: "#taohuayuan" },
    { label: "博客", href: "/blog/" },
    { label: "GitHub", href: "https://github.com/Ethan-a2", external: true },
  ],
  quickLinks: [
    {
      label: "Blog",
      title: "寻访博客",
      href: "/blog/",
      description: "文章、学习记录和长期笔记，如沿溪而行，偶有所得。",
      accent: "leaf",
    },
    {
      label: "GitHub",
      title: "GitHub Profile",
      href: "https://github.com/Ethan-a2",
      description: "开源项目、实验仓库和代码活动，留作阡陌路标。",
      accent: "water",
      external: true,
    },
    {
      label: "Portfolio",
      title: "阡陌作品集",
      href: "#portfolio",
      description: "用于陈列项目、设计稿、演示和案例。",
      accent: "peach",
    },
  ],
  portfolio: [
    {
      name: "Eagle 2.16x Speculative Decoding",
      type: "AI / Performance",
      href: "https://github.com/Ethan-a2/Eagle-2.16x-acceleration-in-5-minutes",
      description: "用一层 Llama 草稿模型实现 2.16x 推测解码加速，并验证逐字一致性。",
    },
    {
      name: "Floor Plan Designer",
      type: "Tool / SVG",
      href: "https://ethan-a2.github.io/Floor-plan-designer/",
      description: "单 HTML 文件实现的 2D 户型设计器，纯 SVG 与原生 JavaScript，零依赖。",
    },
    {
      name: "Player",
      type: "Web App",
      href: "https://ethan-a2.github.io/player",
      description: "在线播放器项目。",
    },
    {
      name: "Stopwatch",
      type: "Web App",
      href: "https://ethan-a2.github.io/stopwatch/",
      description: "在线秒表工具。",
    },
    {
      name: "Ruler",
      type: "Web App",
      href: "https://ethan-a2.github.io/ruler/",
      description: "在线尺子工具。",
    },
  ],
};
