from pydantic import BaseModel, Field

class ThyroidInput(BaseModel):
    TT4: float = Field(..., ge=0, description="Total T4 - مستوى هرمون الغدة الدرقية الكلي (µg/dL). طبيعي ~ 4.5-12.0")
    TSH: float = Field(..., ge=0, description="Thyroid Stimulating Hormone - منشط الغدة الدرقية (µIU/mL). طبيعي ~ 0.4-4.0")
    T3: float = Field(..., ge=0, description="Triiodothyronine - الهرمون النشط للتمثيل الغذائي (ng/mL). طبيعي ~ 0.8-2.0")
    FTI: float = Field(..., ge=0, description="Free Thyroxine Index - مؤشر T4 الحر، يعكس كمية T4 الفعالة")
    T4U: float = Field(..., ge=0, description="T4 Uptake - قدرة البروتينات على ربط T4")
    age: int = Field(..., ge=0, le=120, description="عمر المريض بالسنوات")
    on_thyroxine: int = Field(..., ge=0, le=1, description="هل المريض بياخد دواء Thyroxine؟ 0=لا, 1=نعم")
    thyroid_surgery: int = Field(..., ge=0, le=1, description="هل المريض عمل جراحة للغدة الدرقية؟ 0=لا, 1=نعم")
    query_hyperthyroid: int = Field(..., ge=0, le=1, description="هل فيه شك في فرط نشاط الغدة؟ 0=لا, 1=نعم")
