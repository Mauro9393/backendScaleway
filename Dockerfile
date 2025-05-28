FROM node:18-alpine
WORKDIR /app

# Installa le dipendenze
COPY package*.json ./
RUN npm ci --only=production

# Copia il resto del codice
COPY . .

# Esponi la porta
ENV PORT=8080
EXPOSE 8080

# Comando di avvio
CMD ["node", "server.js"]