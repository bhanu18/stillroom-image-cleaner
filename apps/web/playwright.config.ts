import {defineConfig} from '@playwright/test';
export default defineConfig({testDir:'./tests',workers:1,use:{baseURL:'http://127.0.0.1:8765',viewport:{width:1440,height:1000}},webServer:{command:'../../.runtime/bin/python ../../scripts/e2e_server.py',url:'http://127.0.0.1:8765/health/live',reuseExistingServer:false,timeout:30000}});
