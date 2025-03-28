ps aux | grep ray::IDLE | grep -v grep | awk '{print $2}' | xargs kill -9
