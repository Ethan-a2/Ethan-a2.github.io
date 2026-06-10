window.siteConfig = {
  owner: {
    name: "Ethan-a2",
    handle: "@Ethan-a2",
    tagline: "个人主页 · 博客入口 · 作品集导航",
    intro:
      "这里是主域名总入口。把博客、项目、作品集和常用链接集中到一个 404 灵感的个人导航页。",
  },
  navigation: [
    { label: "博客", href: "/blog/" },
    { label: "作品集", href: "#portfolio" },
    { label: "链接", href: "#links" },
    { label: "GitHub", href: "https://github.com/Ethan-a2", external: true },
  ],
  quickLinks: [
    {
      label: "Blog",
      title: "博客 / Notes",
      href: "/blog/",
      description: "文章、学习记录和长期笔记的入口。",
      accent: "cyan",
    },
    {
      label: "GitHub",
      title: "GitHub Profile",
      href: "https://github.com/Ethan-a2",
      description: "开源项目、实验仓库和代码活动。",
      accent: "violet",
      external: true,
    },
    {
      label: "Portfolio",
      title: "作品集索引",
      href: "#portfolio",
      description: "用于陈列项目、设计稿、演示和案例。",
      accent: "green",
    },
  ],
  portfolio: [
    {
      name: "作品集占位 A",
      type: "Project",
      href: "#",
      description: "替换为你的项目地址、在线 Demo 或案例页面。",
    },
    {
      name: "作品集占位 B",
      type: "Design",
      href: "#",
      description: "适合放设计作品、页面复刻或视觉实验。",
    },
    {
      name: "作品集占位 C",
      type: "Lab",
      href: "#",
      description: "适合放工具、脚本、小应用和学习实验。",
    },
  ],
};
