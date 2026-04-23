# Nginx 反向代理配置示例

> 当前目录暂未提供现成配置文件，以下是一份推荐的生产环境 Nginx 反向代理模板，可根据实际域名和端口调整。

## 用途

- 将外部 80/443 端口统一转发到本地 `qwen3-asr-toolkit` 服务（默认 18000）
- 提供 WebSocket 升级支持（`/ws/` 路径）
- 建议配合 SSL/TLS 证书使用

## 基础配置模板

```nginx
upstream qwen3_asr_backend {
    server 127.0.0.1:18000;
    keepalive 32;
}

server {
    listen 80;
    server_name asr.your-domain.com;

    # 可选：强制跳转 HTTPS
    # return 301 https://$server_name$request_uri;

    location / {
        proxy_pass http://qwen3_asr_backend;
        proxy_http_version 1.1;

        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # 长连接与超时
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 300s;
    }

    location /ws/ {
        proxy_pass http://qwen3_asr_backend;
        proxy_http_version 1.1;

        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;

        # WebSocket 需要更长的超时
        proxy_read_timeout 86400s;
        proxy_send_timeout 60s;
    }
}
```

## 启用配置

```bash
# 假设已将上述内容保存为 /etc/nginx/sites-available/qwen3-asr
sudo ln -s /etc/nginx/sites-available/qwen3-asr /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

## HTTPS 建议

生产环境强烈建议启用 HTTPS。可使用 Let's Encrypt：

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d asr.your-domain.com
```

## 参考

- 服务部署详情：[doc/DEPLOYMENT.md](../../doc/DEPLOYMENT.md)
- systemd 托管配置：[deploy/systemd/qwen3-asr-toolkit.service](../systemd/qwen3-asr-toolkit.service)
