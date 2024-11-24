from dataclasses import dataclass, field
from typing import Dict, Any
from datetime import datetime

@dataclass
class LabelData:
    label: str
    confidence: str
    timestamp: str
    version: int = 1

    def to_dict(self) -> Dict:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
            "version": self.version
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'LabelData':
        return cls(
            label=data['label'],
            confidence=data['confidence'],
            timestamp=data['timestamp'],
            version=data.get('version', 1)
        )

@dataclass
class CategoryProgress:
    total_labeled: int = 0
    last_labeled_index: int = -1
    last_activity: datetime = field(default_factory=datetime.utcnow)
    labels: Dict[str, LabelData] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "total_labeled": self.total_labeled,
            "last_labeled_index": self.last_labeled_index,
            "last_activity": self.last_activity.isoformat(),
            "labels": {k: v.to_dict() for k, v in self.labels.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'CategoryProgress':
        return cls(
            total_labeled=data.get('total_labeled', 0),
            last_labeled_index=data.get('last_labeled_index', -1),
            last_activity=datetime.fromisoformat(data['last_activity']) if isinstance(data.get('last_activity'), str) else data.get('last_activity', datetime.utcnow()),
            labels={k: LabelData.from_dict(v) for k, v in data.get('labels', {}).items()}
        )

@dataclass
class UserProgress:
    user_id: str
    progress: Dict[str, CategoryProgress] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict) -> 'UserProgress':
        return cls(
            user_id=data['user_id'],
            progress={
                k: CategoryProgress.from_dict(v) for k, v in data.get('progress', {}).items()
            }
        )

    def to_dict(self) -> Dict:
        return {
            "user_id": self.user_id,
            "progress": {k: v.to_dict() for k, v in self.progress.items()}
        }