const mongoose = require('mongoose');

const NotificationSchema = new mongoose.Schema({
  type: {
    type: String,
    required: true,
    enum: ['alert', 'news', 'gain', 'loss', 'info', 'market']
  },
  title: {
    type: String,
    required: true
  },
  message: {
    type: String,
    required: true
  },
  symbol: {
    type: String,
    default: null
  },
  data: {
    type: Object,
    default: {}
  },
  // Global notifications (shown to all users)
  global: {
    type: Boolean,
    default: true
  },
  // Expiry time for auto-cleanup
  expiresAt: {
    type: Date,
    default: () => new Date(Date.now() + 24 * 60 * 60 * 1000) // 24 hours
  }
}, {
  timestamps: true
});

// Index for efficient queries
NotificationSchema.index({ createdAt: -1 });
NotificationSchema.index({ expiresAt: 1 }, { expireAfterSeconds: 0 }); // TTL index for auto-cleanup
NotificationSchema.index({ type: 1, createdAt: -1 });

module.exports = mongoose.model('Notification', NotificationSchema);
