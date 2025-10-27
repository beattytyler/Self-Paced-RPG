from extensions import db
from datetime import datetime
from sqlalchemy import Enum

# ---------------------
# User Model
# ---------------------

class User(db.Model):
    __tablename__ = 'user'

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), nullable=False, unique=True)
    email = db.Column(db.String(100), nullable=False, unique=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(Enum('student', 'teacher', name='user_roles'), nullable=False)
    code = db.Column(db.String(10), nullable=True, unique=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


    # Relationships
    classes = db.relationship('Class', backref='teacher', lazy=True)
    registrations = db.relationship('ClassRegistration', backref='student', lazy=True)

    def __repr__(self):
        return f"<User {self.username} ({self.role})>"

# ---------------------
# Class Model
# ---------------------
class Class(db.Model):
    __tablename__ = 'class'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    code = db.Column(db.String(10), nullable=False, unique=True)
    teacher_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    registrations = db.relationship('ClassRegistration', backref='class_', lazy=True)

    def __repr__(self):
        return f"<Class {self.name} ({self.code})>"

# ---------------------
# ClassRegistration Model
# ---------------------
class ClassRegistration(db.Model):
    __tablename__ = 'class_registration'

    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    class_id = db.Column(db.Integer, db.ForeignKey('class.id'), nullable=False)
    registered_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Ensure a student cannot register for the same class twice
    __table_args__ = (db.UniqueConstraint('student_id', 'class_id', name='_student_class_uc'),)

    def __repr__(self):
        return f"<Registration Student:{self.student_id} Class:{self.class_id}>"
