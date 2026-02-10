const Notification = require('../schemas/Notification');
const { DateTime } = require('luxon');

// Track last generated notifications to avoid duplicates
const lastNotifications = {
  marketOpen: null,
  marketClose: null,
  preMarket: null,
  afterHours: null,
  fearGreed: null,
  priceAlerts: new Map() // symbol -> lastAlertTime
};

// Generate market session notifications
async function checkMarketSessionNotifications() {
  const now = DateTime.now().setZone('America/New_York');
  const today = now.toISODate();
  const hour = now.hour;
  const minute = now.minute;
  const timeInMinutes = hour * 60 + minute;
  const dayOfWeek = now.weekday; // 1=Monday, 7=Sunday

  // Skip weekends
  if (dayOfWeek > 5) return;

  const notifications = [];

  // Pre-market opens at 4:00 AM ET
  if (timeInMinutes >= 240 && timeInMinutes < 245) {
    if (lastNotifications.preMarket !== today) {
      notifications.push({
        type: 'market',
        title: 'Pre-Market Open',
        message: 'Pre-market trading session has started. Early movers may indicate market direction.',
        global: true
      });
      lastNotifications.preMarket = today;
    }
  }

  // Regular market opens at 9:30 AM ET
  if (timeInMinutes >= 570 && timeInMinutes < 575) {
    if (lastNotifications.marketOpen !== today) {
      notifications.push({
        type: 'alert',
        title: 'Market Open',
        message: 'US stock markets are now open for regular trading. Good luck today!',
        global: true
      });
      lastNotifications.marketOpen = today;
    }
  }

  // Regular market closes at 4:00 PM ET
  if (timeInMinutes >= 960 && timeInMinutes < 965) {
    if (lastNotifications.marketClose !== today) {
      notifications.push({
        type: 'alert',
        title: 'Market Closed',
        message: 'Regular trading session has ended. After-hours trading is now active.',
        global: true
      });
      lastNotifications.marketClose = today;
    }
  }

  // After-hours ends at 8:00 PM ET
  if (timeInMinutes >= 1200 && timeInMinutes < 1205) {
    if (lastNotifications.afterHours !== today) {
      notifications.push({
        type: 'market',
        title: 'After-Hours Closed',
        message: 'Extended trading session has ended. Markets will reopen tomorrow.',
        global: true
      });
      lastNotifications.afterHours = today;
    }
  }

  // Save notifications to database
  for (const notif of notifications) {
    try {
      await Notification.create(notif);
      console.log(`📢 Created notification: ${notif.title}`);
    } catch (error) {
      console.error('Error creating notification:', error.message);
    }
  }

  return notifications;
}

// Generate Fear & Greed notifications
async function checkFearGreedNotification(fgiData) {
  if (!fgiData?.fgi?.now?.value) return null;

  const currentValue = fgiData.fgi.now.value;
  const currentText = fgiData.fgi.now.valueText;
  const previousValue = fgiData.fgi.previousClose?.value;

  // Only notify on significant changes (>5 points) or extreme values
  const shouldNotify = 
    (previousValue && Math.abs(currentValue - previousValue) >= 5) ||
    currentValue <= 20 || // Extreme Fear
    currentValue >= 80;   // Extreme Greed

  // Don't spam - only once per hour
  const now = Date.now();
  if (lastNotifications.fearGreed && (now - lastNotifications.fearGreed) < 60 * 60 * 1000) {
    return null;
  }

  if (shouldNotify) {
    let title, message, type;

    if (currentValue <= 20) {
      title = 'Extreme Fear in Market';
      message = `Fear & Greed Index at ${currentValue} (${currentText}). Markets showing extreme fear - potential buying opportunity.`;
      type = 'alert';
    } else if (currentValue >= 80) {
      title = 'Extreme Greed in Market';
      message = `Fear & Greed Index at ${currentValue} (${currentText}). Markets showing extreme greed - consider caution.`;
      type = 'alert';
    } else if (previousValue && currentValue > previousValue) {
      title = 'Market Sentiment Rising';
      message = `Fear & Greed Index moved to ${currentValue} (${currentText}) from ${previousValue}.`;
      type = 'info';
    } else {
      title = 'Market Sentiment Falling';
      message = `Fear & Greed Index dropped to ${currentValue} (${currentText}) from ${previousValue}.`;
      type = 'info';
    }

    try {
      const notif = await Notification.create({
        type,
        title,
        message,
        data: { fearGreedIndex: currentValue, previousValue },
        global: true
      });
      lastNotifications.fearGreed = now;
      console.log(`📢 Created Fear & Greed notification: ${title}`);
      return notif;
    } catch (error) {
      console.error('Error creating Fear & Greed notification:', error.message);
    }
  }

  return null;
}

// Generate price movement notifications
async function checkPriceAlert(symbol, currentPrice, previousPrice, changePercent) {
  if (!symbol || !currentPrice || !previousPrice) return null;

  const absChange = Math.abs(changePercent);
  const now = Date.now();

  // Throttle: max one alert per symbol per 30 minutes
  const lastAlert = lastNotifications.priceAlerts.get(symbol);
  if (lastAlert && (now - lastAlert) < 30 * 60 * 1000) {
    return null;
  }

  // Only alert on significant moves (>3%)
  if (absChange < 3) return null;

  const isGain = changePercent > 0;
  const type = isGain ? 'gain' : 'loss';
  const direction = isGain ? 'up' : 'down';
  const emoji = isGain ? '📈' : '📉';

  const title = `${symbol} ${isGain ? 'Surging' : 'Dropping'}`;
  const message = `${symbol} is ${direction} ${absChange.toFixed(1)}% to $${currentPrice.toFixed(2)}`;

  try {
    const notif = await Notification.create({
      type,
      title,
      message,
      symbol,
      data: { 
        price: currentPrice, 
        previousPrice, 
        changePercent 
      },
      global: true
    });
    lastNotifications.priceAlerts.set(symbol, now);
    console.log(`📢 Created price alert: ${title}`);
    return notif;
  } catch (error) {
    console.error('Error creating price alert:', error.message);
    return null;
  }
}

// Get recent notifications
async function getNotifications(limit = 20, since = null) {
  try {
    const query = {};
    if (since) {
      query.createdAt = { $gt: new Date(since) };
    }

    const notifications = await Notification.find(query)
      .sort({ createdAt: -1 })
      .limit(limit)
      .lean();

    return notifications.map(n => ({
      id: n._id.toString(),
      type: n.type,
      title: n.title,
      message: n.message,
      symbol: n.symbol,
      timestamp: n.createdAt,
      data: n.data
    }));
  } catch (error) {
    console.error('Error fetching notifications:', error.message);
    return [];
  }
}

// Get unread count (notifications in last hour)
async function getUnreadCount(since = null) {
  try {
    const cutoff = since ? new Date(since) : new Date(Date.now() - 60 * 60 * 1000); // Last hour
    const count = await Notification.countDocuments({
      createdAt: { $gt: cutoff }
    });
    return count;
  } catch (error) {
    console.error('Error getting unread count:', error.message);
    return 0;
  }
}

// Cleanup old notifications (called periodically)
async function cleanupOldNotifications() {
  try {
    const cutoff = new Date(Date.now() - 48 * 60 * 60 * 1000); // 48 hours
    const result = await Notification.deleteMany({
      createdAt: { $lt: cutoff }
    });
    if (result.deletedCount > 0) {
      console.log(`🧹 Cleaned up ${result.deletedCount} old notifications`);
    }
  } catch (error) {
    console.error('Error cleaning up notifications:', error.message);
  }
}

module.exports = {
  checkMarketSessionNotifications,
  checkFearGreedNotification,
  checkPriceAlert,
  getNotifications,
  getUnreadCount,
  cleanupOldNotifications
};
