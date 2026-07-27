from __future__ import annotations

import random

from ..domain.models import ContentSeeds, ContextFrame


GENERIC_ROLES = (
    "nhân viên hỗ trợ", "bộ phận kỹ thuật", "nhóm xử lý", "điều phối viên",
    "bộ phận vận hành", "nhóm kiểm duyệt", "nhân viên tiếp nhận", "tổ chuyên môn",
)

ACTIONS = (
    "kiểm tra yêu cầu", "cập nhật trạng thái", "chuyển hồ sơ", "đối chiếu hướng dẫn",
    "xác nhận quy trình", "phân loại nội dung", "hoàn tất bước rà soát", "ghi nhận phản hồi",
)

OBJECTS = (
    "biểu mẫu", "yêu cầu hỗ trợ", "hướng dẫn sử dụng", "phiếu nội bộ",
    "quy trình xử lý", "nội dung tổng hợp", "tài liệu hướng dẫn", "bản ghi nghiệp vụ",
)

DOMAIN_TERMS = {
    "technical_support": ("nhóm vận hành kỹ thuật", "quy trình khắc phục"),
    "travel": ("bộ phận điều phối hành trình", "quy trình đặt chỗ"),
    "finance": ("nhóm kiểm soát nghiệp vụ", "quy trình đối soát"),
    "healthcare": ("tổ tiếp nhận chuyên môn", "quy trình theo dõi"),
    "delivery": ("nhóm điều phối giao nhận", "quy trình phân tuyến"),
}


def build_content_seeds(frame: ContextFrame, rng: random.Random) -> ContentSeeds:
    roles = list(GENERIC_ROLES)
    objects = list(OBJECTS)
    domain_terms = DOMAIN_TERMS.get(frame.domain)
    if domain_terms:
        roles.append(domain_terms[0])
        objects.append(domain_terms[1])
    return ContentSeeds(
        domain=frame.domain,
        document_type=frame.document_type,
        tone=frame.tone,
        generic_roles=rng.sample(roles, 3),
        actions=rng.sample(list(ACTIONS), 3),
        objects=rng.sample(objects, 3),
    )
