const express = require('express');
const bodyParser = require('body-parser');
const cors = require('cors');
const mongoose = require('mongoose');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');

const app = express();
const PORT = process.env.PORT || 5000;
const MONGO_URI = process.env.MONGO_URI || 'mongodb://localhost:27017/finance';
const JWT_SECRET = process.env.JWT_SECRET || 'secret-key';

mongoose.connect(MONGO_URI, {
  useNewUrlParser: true,
  useUnifiedTopology: true
});

const UserSchema = new mongoose.Schema({
  username: { type: String, unique: true },
  password: String
});

const TransactionSchema = new mongoose.Schema({
  description: String,
  amount: Number,
  category: String,
  createdAt: { type: Date, default: Date.now },
  userId: mongoose.Schema.Types.ObjectId
});

const BudgetSchema = new mongoose.Schema({
  month: String,
  amount: Number,
  userId: mongoose.Schema.Types.ObjectId
});

const GoalSchema = new mongoose.Schema({
  name: String,
  target: Number,
  progress: { type: Number, default: 0 },
  userId: mongoose.Schema.Types.ObjectId
});

const User = mongoose.model('User', UserSchema);
const Transaction = mongoose.model('Transaction', TransactionSchema);
const Budget = mongoose.model('Budget', BudgetSchema);
const Goal = mongoose.model('Goal', GoalSchema);

app.use(cors());
app.use(bodyParser.json());

// ----- Authentication -----
function auth(req, res, next) {
  const token = req.headers.authorization?.split(' ')[1];
  if (!token) return res.status(401).json({ message: 'No token provided' });
  try {
    const user = jwt.verify(token, JWT_SECRET);
    req.userId = user.id;
    next();
  } catch (err) {
    res.status(401).json({ message: 'Invalid token' });
  }
}

app.post('/api/register', async (req, res) => {
  const { username, password } = req.body;
  const hashed = await bcrypt.hash(password, 10);
  try {
    const user = await User.create({ username, password: hashed });
    res.status(201).json({ id: user._id, username: user.username });
  } catch {
    res.status(400).json({ message: 'User already exists' });
  }
});

app.post('/api/login', async (req, res) => {
  const { username, password } = req.body;
  const user = await User.findOne({ username });
  if (!user) return res.status(401).json({ message: 'Invalid credentials' });
  const match = await bcrypt.compare(password, user.password);
  if (!match) return res.status(401).json({ message: 'Invalid credentials' });
  const token = jwt.sign({ id: user._id }, JWT_SECRET);
  res.json({ token });
});

// ----- Transactions -----
app.get('/api/transactions', auth, async (req, res) => {
  const items = await Transaction.find({ userId: req.userId });
  res.json(items);
});

app.post('/api/transactions', auth, async (req, res) => {
  const transaction = await Transaction.create({ ...req.body, userId: req.userId });
  res.status(201).json(transaction);
});

app.put('/api/transactions/:id', auth, async (req, res) => {
  const t = await Transaction.findOneAndUpdate({ _id: req.params.id, userId: req.userId }, req.body, { new: true });
  res.json(t);
});

app.delete('/api/transactions/:id', auth, async (req, res) => {
  await Transaction.deleteOne({ _id: req.params.id, userId: req.userId });
  res.status(204).end();
});

// ----- Budgets -----
app.get('/api/budgets', auth, async (req, res) => {
  const items = await Budget.find({ userId: req.userId });
  res.json(items);
});

app.post('/api/budgets', auth, async (req, res) => {
  const budget = await Budget.create({ ...req.body, userId: req.userId });
  res.status(201).json(budget);
});

// ----- Goals -----
app.get('/api/goals', auth, async (req, res) => {
  const items = await Goal.find({ userId: req.userId });
  res.json(items);
});

app.post('/api/goals', auth, async (req, res) => {
  const goal = await Goal.create({ ...req.body, userId: req.userId });
  res.status(201).json(goal);
});

app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
