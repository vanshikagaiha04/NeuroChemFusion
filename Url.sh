cat > ~/start_tunnel.sh << 'EOF'
#!/bin/bash
pkill cloudflared 2>/dev/null
sleep 2
nohup ~/cloudflared tunnel --url ssh://localhost:22 > ~/cloudflared.log 2>&1 &
sleep 8
URL=$(grep -o 'https://[a-z-]*\.trycloudflare\.com' ~/cloudflared.log | tail -1)
echo $URL > ~/tunnel_url.txt
# URL ko terminal pe print karo
echo "=============================="
echo "TUNNEL URL: $URL"
echo "=============================="
echo "Share this with Vanshika!"
EOF
chmod +x ~/start_tunnel.sh