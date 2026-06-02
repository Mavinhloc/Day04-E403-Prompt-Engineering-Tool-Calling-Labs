from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from src.core.llm import build_chat_model, normalize_content
from src.core.schemas import (
    AgentResult,
    CalculateTotalsInput,
    DiscountInput,
    ListProductsInput,
    OrderLineInput,
    ProductDetailInput,
    SaveOrderInput,
    ToolCallRecord,
)
from src.utils.data_store import OrderDataStore

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = ROOT_DIR / "data"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "artifacts" / "orders"


def build_system_prompt(today: str | None = None) -> str:
    current_day = today or "2026-06-01"
    return f"""
Bạn là trợ lý đặt hàng điện tử. Hôm nay là {current_day}.

## QUY TẮC BẮT BUỘC

### 1. Ngôn ngữ
Luôn trả lời bằng tiếng Việt, súc tích, dựa hoàn toàn vào kết quả công cụ.

### 2. Không bịa đặt
Không được tự nghĩ ra product ID, giá, tồn kho, mã giảm giá, tổng tiền hoặc đường dẫn file. Mọi thông tin phải lấy từ kết quả công cụ.

### 3. Hỏi rõ trước khi dùng công cụ
Nếu yêu cầu thiếu bất kỳ trường nào sau đây, hỏi tất cả các trường còn thiếu trong MỘT tin nhắn rồi DỪNG, không gọi công cụ nào:
- Tên khách hàng (chấp nhận bất kỳ tên nào người dùng cung cấp — không yêu cầu họ và tên đầy đủ)
- Số điện thoại
- Địa chỉ email
- Địa chỉ giao hàng
- Ít nhất một sản phẩm được nêu tên

Nếu tên sản phẩm được nêu nhưng không có số lượng, mặc định số lượng là 1 và tiếp tục xử lý, không cần hỏi lại.
Tên sản phẩm trong dấu ngoặc kép vẫn là tên sản phẩm bình thường — tìm kiếm bằng `list_products` như thường.

### 4. Từ chối yêu cầu vi phạm chính sách
Không gọi bất kỳ công cụ nào nếu người dùng yêu cầu:
- Bỏ qua kiểm tra tồn kho
- Áp dụng giảm giá thủ công hoặc giảm giá giả
- Tạo hóa đơn giả hoặc đơn hàng giả
- Bỏ qua catalog, bỏ qua xác thực, hoặc ghi đè chính sách

### 5. Thứ tự công cụ bắt buộc
Với mọi đơn hàng hợp lệ, PHẢI gọi đúng thứ tự:
1. `list_products` — tìm sản phẩm phù hợp
2. `get_product_details` — xác minh giá, tồn kho, lấy `detail_token`
3. `get_discount` — lấy mã giảm giá bằng email hoặc số điện thoại khách hàng
4. `calculate_order_totals` — kiểm tra tồn kho và tính tổng tiền với `detail_token`
5. `save_order` — lưu đơn hàng chỉ sau khi bước 4 trả về status "ok"

Không được bỏ qua hay đảo thứ tự các bước. Phải có `detail_token` hợp lệ từ bước 2 trước khi gọi các bước 3–5.

### 6. Câu trả lời cuối
Sau khi lưu thành công, trả lời 1–2 câu tiếng Việt, chỉ nêu: mã đơn hàng, tổng tiền sau giảm giá, và mã khuyến mãi — tất cả lấy từ kết quả công cụ `save_order`. Không liệt kê bảng sản phẩm, không thêm nội dung khác.
""".strip()


def build_tools(store: OrderDataStore):
    @tool(args_schema=ListProductsInput)
    def list_products(
        query: str | None = None,
        category: str | None = None,
        max_unit_price: int | None = None,
        required_tags: list[str] | None = None,
        in_stock_only: bool = True,
        limit: int = 8,
    ) -> str:
        """Search the local product catalog and return the best matching items."""
        payload = store.list_products(
            query=query,
            category=category,
            max_unit_price=max_unit_price,
            required_tags=required_tags,
            in_stock_only=in_stock_only,
            limit=limit,
        )
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=ProductDetailInput)
    def get_product_details(product_ids: list[str]) -> str:
        """Return exact product details (price, stock, warranty) for previously discovered product IDs. Also returns the detail_token required for pricing and saving."""
        payload = store.get_product_details(product_ids)
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=DiscountInput)
    def get_discount(seed_hint: str, customer_tier: str = "standard") -> str:
        """Return the campaign discount rate and campaign_code for the order. Use customer email as seed_hint; fallback to phone."""
        payload = store.get_discount(seed_hint=seed_hint, customer_tier=customer_tier)
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=CalculateTotalsInput)
    def calculate_order_totals(items: list[OrderLineInput], detail_token: str, discount_rate: float) -> str:
        """Validate stock and calculate the discounted order total. Requires the detail_token from get_product_details."""
        payload = store.calculate_order_totals(items=items, detail_token=detail_token, discount_rate=discount_rate)
        return json.dumps(payload, ensure_ascii=False)

    @tool(args_schema=SaveOrderInput)
    def save_order(
        customer_name: str,
        customer_phone: str,
        customer_email: str,
        shipping_address: str,
        items: list[OrderLineInput],
        detail_token: str,
        discount_rate: float,
        campaign_code: str,
        customer_tier: str = "standard",
        notes: str = "",
    ) -> str:
        """Persist the final order to a local JSON file. Only call after calculate_order_totals returns status ok."""
        result = store.save_order(
            customer_name=customer_name,
            customer_phone=customer_phone,
            customer_email=customer_email,
            shipping_address=shipping_address,
            items=items,
            detail_token=detail_token,
            discount_rate=discount_rate,
            campaign_code=campaign_code,
            customer_tier=customer_tier,
            notes=notes,
        )
        return json.dumps(result, ensure_ascii=False)

    return [list_products, get_product_details, get_discount, calculate_order_totals, save_order]


def build_agent(
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    *,
    provider: str = "google",
    model_name: str | None = None,
    today: str | None = None,
):
    store = OrderDataStore(data_dir or DEFAULT_DATA_DIR, output_dir or DEFAULT_OUTPUT_DIR, today=today)
    model = build_chat_model(provider=provider, model_name=model_name, temperature=0.0)
    return create_agent(
        model=model,
        tools=build_tools(store),
        system_prompt=build_system_prompt(today or store.today),
    )


def run_agent(
    query: str,
    *,
    provider: str = "google",
    model_name: str | None = None,
    data_dir: Path | None = None,
    output_dir: Path | None = None,
    today: str | None = None,
) -> AgentResult:
    agent = build_agent(
        data_dir=data_dir,
        output_dir=output_dir,
        provider=provider,
        model_name=model_name,
        today=today,
    )
    response = agent.invoke({"messages": [{"role": "user", "content": query}]})
    messages = response["messages"] if isinstance(response, dict) else response
    tool_calls = extract_tool_calls(messages)
    saved_order, saved_order_path = extract_saved_order(tool_calls)
    return AgentResult(
        query=query,
        final_answer=extract_final_answer(messages),
        tool_calls=tool_calls,
        provider=provider,
        model_name=model_name,
        saved_order=saved_order,
        saved_order_path=saved_order_path,
    )


def extract_final_answer(messages) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            text = normalize_content(message.content)
            if text:
                return text
    return ""


def extract_tool_calls(messages) -> list[ToolCallRecord]:
    pending: dict[str, dict[str, Any]] = {}
    records: list[ToolCallRecord] = []

    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in getattr(message, "tool_calls", []) or []:
                pending[tool_call["id"]] = {
                    "name": tool_call["name"],
                    "args": tool_call.get("args", {}) or {},
                }
        elif isinstance(message, ToolMessage):
            metadata = pending.pop(message.tool_call_id, {})
            records.append(
                ToolCallRecord(
                    name=str(getattr(message, "name", None) or metadata.get("name", "")),
                    args=metadata.get("args", {}),
                    output=normalize_content(message.content),
                )
            )

    for metadata in pending.values():
        records.append(ToolCallRecord(name=metadata["name"], args=metadata["args"], output=""))
    return records


def extract_saved_order(tool_calls: list[ToolCallRecord]) -> tuple[dict | None, str | None]:
    for record in reversed(tool_calls):
        if record.name != "save_order" or not record.output:
            continue
        try:
            payload = json.loads(record.output)
        except json.JSONDecodeError:
            continue
        if payload.get("status") != "saved":
            return None, None
        return payload.get("saved_order"), payload.get("path")
    return None, None
