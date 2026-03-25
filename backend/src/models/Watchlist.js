const mongoose = require('mongoose');

const watchlistSchema = new mongoose.Schema({
  userId: {
    type: String,
    required: true,
    unique: true,
    index: true,
  },
  symbols: {
    type: [String],
    default: [],
    validate: {
      validator: (arr) => arr.length <= 50,
      message: 'Watchlist cannot exceed 50 symbols',
    },
  },
}, {
  timestamps: true, // adds createdAt, updatedAt
  collection: 'user_watchlists',
});

// Ensure symbols are always uppercase
watchlistSchema.pre('save', function (next) {
  this.symbols = this.symbols.map(s => s.toUpperCase());
  next();
});

const Watchlist = mongoose.model('Watchlist', watchlistSchema);

module.exports = Watchlist;
