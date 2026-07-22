FROM node:20-alpine

WORKDIR /app

# Install dependencies into the image's own /app/node_modules first, keyed
# off the lockfile, so this layer is cached across source-only rebuilds.
COPY package.json package-lock.json ./
RUN npm ci

# docker-compose.yml then bind-mounts ./frontend:/app over this layer for
# live-editing, while keeping /app/node_modules and /app/.next as separate
# anonymous volumes so the host bind mount never shadows the
# container-installed dependencies or build cache.
COPY . .

EXPOSE 3000

CMD ["npm", "run", "dev"]
