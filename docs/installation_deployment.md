# ECG Analysis Suite - Installation & Deployment Guide

This document provides detailed instructions for deploying the ECG Analysis Suite on a remote server with Nginx for subdomain routing.

## System Requirements

- Ubuntu 20.04 LTS or newer (or equivalent Linux distribution)
- Python 3.7+ 
- Nginx
- 4GB+ RAM recommended
- 50GB+ storage (depending on expected data volume)

## 1. Server Preparation

### Update System Packages

```bash
sudo apt update
sudo apt upgrade -y
```

### Install Required System Dependencies

```bash
sudo apt install -y python3-pip python3-dev build-essential libssl-dev libffi-dev python3-setuptools python3-venv nginx git
```

## 2. Application Setup

### Clone Repository

```bash
# Create application directory
mkdir -p /opt/ecg
cd /opt/ecg

# Clone the repository 
git clone https://github.com/chirag6451/ecg-analysis.git .
```

### Set Up Python Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate environment
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Install optional performance improvement
pip install watchdog
```

## 3. Configuration

### Create Environment Variables (Optional)

If your application requires environment variables:

```bash
cat > /opt/ecg/.env << EOF
# App Configuration
APP_ENVIRONMENT=production
DEBUG=False

# Other configurations as needed
EOF
```

## 4. Service Setup

Create systemd service files for each application in the suite.

### Main Launcher Service

```bash
sudo nano /etc/systemd/system/ecg-launcher.service
```

Add the following content:

```ini
[Unit]
Description=ECG Analysis Suite Launcher
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/opt/ecg
Environment="PATH=/opt/ecg/venv/bin"
ExecStart=/opt/ecg/venv/bin/streamlit run ecg_launcher.py --server.port 8515 --server.headless true --server.enableCORS false --server.enableXsrfProtection false
Restart=always
RestartSec=5
SyslogIdentifier=ecg-launcher

[Install]
WantedBy=multi-user.target
```

### Application Services (Optional)

If you want to run each application as a separate service:

```bash
sudo nano /etc/systemd/system/ecg-af-detection.service
```

```ini
[Unit]
Description=AFDx: Advanced ECG & AF Detection
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/opt/ecg
Environment="PATH=/opt/ecg/venv/bin"
ExecStart=/opt/ecg/venv/bin/streamlit run af_detection_app.py --server.port 8521 --server.headless true --server.enableCORS false --server.enableXsrfProtection false
Restart=always
RestartSec=5
SyslogIdentifier=ecg-af-detection

[Install]
WantedBy=multi-user.target
```

Create similar services for other applications if needed.

### Enable and Start Services

```bash
# Set proper permissions
sudo chown -R www-data:www-data /opt/ecg

# Enable and start the services
sudo systemctl daemon-reload
sudo systemctl enable ecg-launcher.service
sudo systemctl start ecg-launcher.service

# Optional: Enable and start individual app services
# sudo systemctl enable ecg-af-detection.service
# sudo systemctl start ecg-af-detection.service
```

## 5. Nginx Configuration

### Create Nginx Server Block

```bash
sudo nano /etc/nginx/sites-available/ecg.conf
```

Add the following configuration:

```nginx
server {
    listen 80;
    server_name ecg.yourdomain.com;

    location / {
        proxy_pass http://localhost:8515;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_buffering off;
    }
}

# Optional: Subdomain configuration for individual apps
server {
    listen 80;
    server_name afdx.yourdomain.com;

    location / {
        proxy_pass http://localhost:8521;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_buffering off;
    }
}

server {
    listen 80;
    server_name dashboard.yourdomain.com;

    location / {
        proxy_pass http://localhost:8522;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_buffering off;
    }
}

server {
    listen 80;
    server_name deepdive.yourdomain.com;

    location / {
        proxy_pass http://localhost:8523;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_buffering off;
    }
}

server {
    listen 80;
    server_name holter.yourdomain.com;

    location / {
        proxy_pass http://localhost:8524;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_buffering off;
    }
}
```

### Enable Nginx Configuration

```bash
sudo ln -s /etc/nginx/sites-available/ecg.conf /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

## 6. SSL Setup with Let's Encrypt (Recommended)

### Install Certbot

```bash
sudo apt install -y certbot python3-certbot-nginx
```

### Obtain SSL Certificates

```bash
sudo certbot --nginx -d ecg.yourdomain.com
# Optional: Add SSL to app subdomains
sudo certbot --nginx -d afdx.yourdomain.com -d dashboard.yourdomain.com -d deepdive.yourdomain.com -d holter.yourdomain.com
```

## 7. Firewall Configuration

```bash
sudo ufw allow 'Nginx Full'
sudo ufw enable
```

## 8. Monitoring and Logs

### View Service Logs

```bash
# View launcher logs
sudo journalctl -u ecg-launcher -f

# View individual app logs
sudo journalctl -u ecg-af-detection -f
```

### Nginx Logs

```bash
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

## 9. Maintenance

### Updating the Application

```bash
cd /opt/ecg
source venv/bin/activate

# Pull latest changes
git pull

# Update dependencies if needed
pip install -r requirements.txt

# Restart services
sudo systemctl restart ecg-launcher.service
```

### Backup Configuration

```bash
# Backup Nginx configuration
sudo cp /etc/nginx/sites-available/ecg.conf /etc/nginx/sites-available/ecg.conf.backup

# Backup systemd service files
sudo cp /etc/systemd/system/ecg-launcher.service /etc/systemd/system/ecg-launcher.service.backup
```

## Troubleshooting

### Common Issues and Solutions

1. **Streamlit app not starting:**
   ```bash
   sudo journalctl -u ecg-launcher -f
   ```
   Check for Python errors or dependency issues.

2. **Nginx proxy not working:**
   ```bash
   sudo nginx -t
   ```
   Verify configuration syntax.

3. **Permission issues:**
   ```bash
   sudo chown -R www-data:www-data /opt/ecg
   sudo chmod -R 755 /opt/ecg
   ```
   Ensure proper permissions.

4. **Service not starting automatically:**
   ```bash
   sudo systemctl enable ecg-launcher.service
   sudo systemctl daemon-reload
   ```
   Verify systemd configuration.

## Security Considerations

1. Consider adding authentication to your Streamlit apps
2. Regularly update your server and dependencies
3. Use UFW to restrict access to necessary ports only
4. Consider implementing fail2ban for SSH protection

---

 2025 Chirag Kansara/Ahmedabadi, IndaPoint Technologies Private Limited.
Licensed under MIT License. See LICENSE file for details.