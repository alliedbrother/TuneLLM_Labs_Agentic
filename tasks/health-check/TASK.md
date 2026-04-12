---
name: System Health Check
assignee: infra-management
project: infra-platform
schedule:
  timezone: America/New_York
  startsAt: 2026-03-21T08:00:00-04:00
  recurrence:
    frequency: daily
    interval: 1
    time:
      hour: 8
      minute: 0
---

Daily infrastructure health check. The Infra Management Agent should:

1. Verify all inference serving instances are healthy
2. Check GPU utilization and instance health
3. Review storage usage and capacity
4. Check for any cost anomalies
5. Report any issues to the CEO
