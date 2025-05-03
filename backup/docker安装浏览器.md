docker-compose.yml:
```
services:
  firefox:
    image: jlesage/firefox
    container_name: firefox
    ports:
      - "5800:5800" # 格式: "宿主机端口:容器端口"
    volumes:
      - /media/docker/firefox/config:/config:rw 
      - /media/docker/firefox/downloads:/downloads:rw
    restart: unless-stopped # 可选：添加重启策略，这样容器意外退出或 Docker 重启时会自动启动
```


# references
- https://github.com/jlesage/docker-firefox
- 