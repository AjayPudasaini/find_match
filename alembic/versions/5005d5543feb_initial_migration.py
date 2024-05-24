"""Initial migration

Revision ID: 5005d5543feb
Revises: 
Create Date: 2024-05-24 12:55:39.372054

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '5005d5543feb'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('related_person_similarity',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('primary_data_id', sa.Integer(), nullable=True),
    sa.Column('similar_data_id', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
    sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_related_person_similarity_id'), 'related_person_similarity', ['id'], unique=False)
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index(op.f('ix_related_person_similarity_id'), table_name='related_person_similarity')
    op.drop_table('related_person_similarity')
    # ### end Alembic commands ###