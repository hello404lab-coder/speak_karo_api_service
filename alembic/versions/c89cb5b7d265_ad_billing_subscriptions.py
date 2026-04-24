"""ad_billing_subscriptions

Revision ID: c89cb5b7d265
Revises: 3e8503e5e3b9
Create Date: 2026-04-23 11:41:34.677595

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c89cb5b7d265'
down_revision: Union[str, None] = '3e8503e5e3b9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(inspector: sa.Inspector, table_name: str) -> bool:
    return table_name in inspector.get_table_names()


def _column_exists(inspector: sa.Inspector, table_name: str, column_name: str) -> bool:
    if not _table_exists(inspector, table_name):
        return False
    return any(col["name"] == column_name for col in inspector.get_columns(table_name))


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if not _table_exists(inspector, "billing_subscriptions"):
        op.create_table(
            "billing_subscriptions",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("user_id", sa.String(length=36), nullable=False),
            sa.Column("provider", sa.String(length=32), nullable=False),
            sa.Column("plan_code", sa.String(length=32), nullable=False),
            sa.Column("provider_plan_id", sa.String(length=64), nullable=False),
            sa.Column("provider_subscription_id", sa.String(length=64), nullable=False),
            sa.Column("provider_customer_id", sa.String(length=64), nullable=True),
            sa.Column("provider_payment_id", sa.String(length=64), nullable=True),
            sa.Column("provider_offer_id", sa.String(length=64), nullable=True),
            sa.Column("status", sa.String(length=32), nullable=False),
            sa.Column("billing_phase", sa.String(length=32), nullable=False, server_default=sa.text("'free'")),
            sa.Column("short_url", sa.Text(), nullable=True),
            sa.Column("current_start_at", sa.DateTime(), nullable=True),
            sa.Column("current_end_at", sa.DateTime(), nullable=True),
            sa.Column("charge_at", sa.DateTime(), nullable=True),
            sa.Column("start_at", sa.DateTime(), nullable=True),
            sa.Column("end_at", sa.DateTime(), nullable=True),
            sa.Column("expire_by", sa.DateTime(), nullable=True),
            sa.Column("trial_access_until", sa.DateTime(), nullable=True),
            sa.Column("authenticated_at", sa.DateTime(), nullable=True),
            sa.Column("cancel_at_cycle_end", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("cancelled_at", sa.DateTime(), nullable=True),
            sa.Column("ended_at", sa.DateTime(), nullable=True),
            sa.Column("coupon_id", sa.String(length=36), nullable=True),
            sa.Column("coupon_code_snapshot", sa.String(length=64), nullable=True),
            sa.Column("last_invoice_id", sa.String(length=64), nullable=True),
            sa.Column("last_payment_id", sa.String(length=64), nullable=True),
            sa.Column("raw_last_payload", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("provider_subscription_id"),
        )
        op.create_index(op.f("ix_billing_subscriptions_user_id"), "billing_subscriptions", ["user_id"], unique=False)
        op.create_index(op.f("ix_billing_subscriptions_provider"), "billing_subscriptions", ["provider"], unique=False)
        op.create_index(op.f("ix_billing_subscriptions_plan_code"), "billing_subscriptions", ["plan_code"], unique=False)
        op.create_index(op.f("ix_billing_subscriptions_provider_plan_id"), "billing_subscriptions", ["provider_plan_id"], unique=False)
        op.create_index(op.f("ix_billing_subscriptions_provider_subscription_id"), "billing_subscriptions", ["provider_subscription_id"], unique=True)
        op.create_index(op.f("ix_billing_subscriptions_provider_customer_id"), "billing_subscriptions", ["provider_customer_id"], unique=False)
        op.create_index(op.f("ix_billing_subscriptions_provider_payment_id"), "billing_subscriptions", ["provider_payment_id"], unique=False)
        op.create_index(op.f("ix_billing_subscriptions_provider_offer_id"), "billing_subscriptions", ["provider_offer_id"], unique=False)
        op.create_index(op.f("ix_billing_subscriptions_status"), "billing_subscriptions", ["status"], unique=False)
        op.create_index(op.f("ix_billing_subscriptions_billing_phase"), "billing_subscriptions", ["billing_phase"], unique=False)
        op.create_index(op.f("ix_billing_subscriptions_coupon_id"), "billing_subscriptions", ["coupon_id"], unique=False)
        op.create_index(op.f("ix_billing_subscriptions_coupon_code_snapshot"), "billing_subscriptions", ["coupon_code_snapshot"], unique=False)
        op.create_index(op.f("ix_billing_subscriptions_last_invoice_id"), "billing_subscriptions", ["last_invoice_id"], unique=False)
        op.create_index(op.f("ix_billing_subscriptions_last_payment_id"), "billing_subscriptions", ["last_payment_id"], unique=False)
    else:
        # Table existed from an older schema (e.g. partial / hand migration): align columns
        inspector = sa.inspect(bind)
        if not _column_exists(
            inspector, "billing_subscriptions", "provider_offer_id"
        ):
            op.add_column(
                "billing_subscriptions",
                sa.Column("provider_offer_id", sa.String(length=64), nullable=True),
            )
            op.create_index(
                op.f("ix_billing_subscriptions_provider_offer_id"),
                "billing_subscriptions",
                ["provider_offer_id"],
                unique=False,
            )

    inspector = sa.inspect(bind)
    if not _table_exists(inspector, "billing_coupons"):
        op.create_table(
            "billing_coupons",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("code", sa.String(length=64), nullable=False),
            sa.Column("name", sa.String(length=255), nullable=False),
            sa.Column("status", sa.String(length=32), nullable=False, server_default=sa.text("'enabled'")),
            sa.Column("mode", sa.String(length=32), nullable=False),
            sa.Column("applies_to_plan_code", sa.String(length=32), nullable=False),
            sa.Column("free_cycles", sa.Integer(), nullable=True),
            sa.Column("discount_type", sa.String(length=32), nullable=True),
            sa.Column("discount_value", sa.Integer(), nullable=True),
            sa.Column("max_redemptions_total", sa.Integer(), nullable=True),
            sa.Column("max_redemptions_per_user", sa.Integer(), nullable=False, server_default=sa.text("1")),
            sa.Column("starts_at", sa.DateTime(), nullable=True),
            sa.Column("ends_at", sa.DateTime(), nullable=True),
            sa.Column("razorpay_offer_id", sa.String(length=64), nullable=True),
            sa.Column("metadata_json", sa.JSON(), nullable=True),
            sa.Column("raw_config", sa.JSON(), nullable=True),
            sa.Column("created_by_admin_id", sa.String(length=36), nullable=True),
            sa.Column("updated_by_admin_id", sa.String(length=36), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(["created_by_admin_id"], ["admins.id"]),
            sa.ForeignKeyConstraint(["updated_by_admin_id"], ["admins.id"]),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("code"),
        )
        op.create_index(op.f("ix_billing_coupons_code"), "billing_coupons", ["code"], unique=True)
        op.create_index(op.f("ix_billing_coupons_status"), "billing_coupons", ["status"], unique=False)
        op.create_index(op.f("ix_billing_coupons_mode"), "billing_coupons", ["mode"], unique=False)
        op.create_index(op.f("ix_billing_coupons_applies_to_plan_code"), "billing_coupons", ["applies_to_plan_code"], unique=False)
        op.create_index(op.f("ix_billing_coupons_razorpay_offer_id"), "billing_coupons", ["razorpay_offer_id"], unique=False)
        op.create_index(op.f("ix_billing_coupons_created_by_admin_id"), "billing_coupons", ["created_by_admin_id"], unique=False)
        op.create_index(op.f("ix_billing_coupons_updated_by_admin_id"), "billing_coupons", ["updated_by_admin_id"], unique=False)

    inspector = sa.inspect(bind)
    if not _table_exists(inspector, "billing_coupon_redemptions"):
        op.create_table(
            "billing_coupon_redemptions",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("coupon_id", sa.String(length=36), nullable=False),
            sa.Column("user_id", sa.String(length=36), nullable=False),
            sa.Column("billing_subscription_id", sa.String(length=36), nullable=True),
            sa.Column("status", sa.String(length=32), nullable=False, server_default=sa.text("'reserved'")),
            sa.Column("coupon_code_snapshot", sa.String(length=64), nullable=False),
            sa.Column("effect_snapshot", sa.JSON(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("verified_at", sa.DateTime(), nullable=True),
            sa.Column("consumed_at", sa.DateTime(), nullable=True),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(["coupon_id"], ["billing_coupons.id"]),
            sa.ForeignKeyConstraint(["user_id"], ["users.id"]),
            sa.ForeignKeyConstraint(["billing_subscription_id"], ["billing_subscriptions.id"]),
            sa.PrimaryKeyConstraint("id"),
        )
        op.create_index(op.f("ix_billing_coupon_redemptions_coupon_id"), "billing_coupon_redemptions", ["coupon_id"], unique=False)
        op.create_index(op.f("ix_billing_coupon_redemptions_user_id"), "billing_coupon_redemptions", ["user_id"], unique=False)
        op.create_index(op.f("ix_billing_coupon_redemptions_billing_subscription_id"), "billing_coupon_redemptions", ["billing_subscription_id"], unique=False)
        op.create_index(op.f("ix_billing_coupon_redemptions_status"), "billing_coupon_redemptions", ["status"], unique=False)
        op.create_index(op.f("ix_billing_coupon_redemptions_coupon_code_snapshot"), "billing_coupon_redemptions", ["coupon_code_snapshot"], unique=False)

    inspector = sa.inspect(bind)
    if not _table_exists(inspector, "billing_webhook_events"):
        op.create_table(
            "billing_webhook_events",
            sa.Column("id", sa.String(length=36), nullable=False),
            sa.Column("provider", sa.String(length=32), nullable=False),
            sa.Column("event_type", sa.String(length=64), nullable=False),
            sa.Column("delivery_hash", sa.String(length=64), nullable=False),
            sa.Column("provider_subscription_id", sa.String(length=64), nullable=True),
            sa.Column("payload", sa.JSON(), nullable=False),
            sa.Column("processed_at", sa.DateTime(), nullable=True),
            sa.Column("processing_status", sa.String(length=32), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.PrimaryKeyConstraint("id"),
            sa.UniqueConstraint("delivery_hash"),
        )
        op.create_index(op.f("ix_billing_webhook_events_provider"), "billing_webhook_events", ["provider"], unique=False)
        op.create_index(op.f("ix_billing_webhook_events_event_type"), "billing_webhook_events", ["event_type"], unique=False)
        op.create_index(op.f("ix_billing_webhook_events_delivery_hash"), "billing_webhook_events", ["delivery_hash"], unique=True)
        op.create_index(op.f("ix_billing_webhook_events_provider_subscription_id"), "billing_webhook_events", ["provider_subscription_id"], unique=False)
        op.create_index(op.f("ix_billing_webhook_events_processing_status"), "billing_webhook_events", ["processing_status"], unique=False)


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)

    if _table_exists(inspector, "billing_coupon_redemptions"):
        op.drop_table("billing_coupon_redemptions")
    inspector = sa.inspect(bind)
    if _table_exists(inspector, "billing_coupons"):
        op.drop_table("billing_coupons")
    inspector = sa.inspect(bind)
    if _table_exists(inspector, "billing_webhook_events"):
        op.drop_table("billing_webhook_events")
    inspector = sa.inspect(bind)
    if _table_exists(inspector, "billing_subscriptions"):
        op.drop_table("billing_subscriptions")
