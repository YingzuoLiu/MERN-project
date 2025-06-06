# Personal Finance Management Application

This repository contains a minimal MERN (MongoDB, Express, React, Node) style application for tracking personal finances. It now persists data to MongoDB and includes authentication. It provides:

- **Income/expense records** with category support
- **Monthly budget** setup and tracking
- **Simple UI** with basic chart support
- **Financial goals** and progress tracking

## Structure

- `server/` – Express API serving transaction, budget and goal endpoints with JWT-based user authentication
- `client/` – Static React front-end using CDN scripts and Chart.js for simple visualization

## Setup

1. Install dependencies in both `server` and `client` directories:
   ```bash
   cd server && npm install
   cd ../client && npm install
   ```
   (Network access is required for this step.)

2. Start the API server (MongoDB must be running locally or specify `MONGO_URI`):
   ```bash
   cd server
   export MONGO_URI="your-mongo-uri"
   export JWT_SECRET="your-secret"
   npm start
   ```
   The server listens on port `5000`.

3. In a separate terminal start the client:
   ```bash
   cd client
   npm start
   ```
   Then open `http://localhost:3000` in your browser.

This project remains intentionally lightweight. For a real deployment you would enable HTTPS and configure environment variables for the database URI and JWT secret.
