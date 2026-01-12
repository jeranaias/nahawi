# Nahawi Model Gap Analysis Report
## Two-Pass Correction Test on 100 Sentences

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Total Test Sentences** | 100 |
| **Total Errors in Original** | ~350-400 |
| **Pass 1 Corrections** | ~180-200 (~50%) |
| **Pass 2 Additional Corrections** | ~15-20 (~4%) |
| **Remaining Uncorrected** | ~150-180 (~40-45%) |

---

## Pass 1 Analysis (Original → First Correction)

### Successfully Corrected

| Error Type | Examples | Approx Count | Success Rate |
|------------|----------|--------------|--------------|
| **Taa Marbuta (ه→ة)** | دراسه→دراسة, الحديثه→الحديثة | ~60-70 | ~75% |
| **Hamza (أ/إ/ء)** | اعلنت→أعلنت, الى→إلى, اكثر→أكثر | ~40-50 | ~70% |
| **ذ/د Confusion** | هده→هذه, هدا→هذا, الدي→الذي | ~15-18 | ~85% |
| **ض/ظ Confusion** | نضر→نظر, انتضر→انتظر, ملحوض→ملحوظ | ~8-10 | ~80% |
| **Other Spelling** | معضم→معظم, المنضمات→المنظمات | ~10-12 | ~70% |

### Missed in Pass 1

| Error Type | Examples Still Wrong | Issue |
|------------|---------------------|-------|
| **First Sentence** | اعلنت الحكومه خطه جديده البنيه التحتيه الريفيه القادمه | Complete miss - 8+ errors |
| **Gender Agreement** | الطالبات المتفوقين (should be المتفوقات) | Grammar not learned |
| **Number Agreement** | الموظفين يحتاجوا (should be يحتاجون) | Grammar not learned |
| **Scattered Taa Marbuta** | الوزاره, الفتره, المكتبه, تربيه | Inconsistent |

---

## Pass 2 Analysis (First Correction → Second Correction)

### Additional Corrections in Pass 2

| Correction | Before | After | Type |
|------------|--------|-------|------|
| زاويه مختلفه | زاويه مختلفه | زاوية مختلفة | Taa Marbuta |
| انحاء | انحاء | أنحاء | Hamza |
| تحتوى | تحتوى | تحتوي | Alif Maqsura |
| مبتكره | مبتكره | مبتكرة | Taa Marbuta |
| المتجدده | المتجدده | المتجددة | Taa Marbuta |
| Added periods | (none) | . | Punctuation |

### Still Uncorrected After 2 Passes

#### 1. First Sentence (Critical Bug)
```
STILL WRONG: اعلنت الحكومه عن خطه جديده لتطوير البنيه التحتيه في المناطق الريفيه خلال السنوات الخمس القادمه

SHOULD BE:  أعلنت الحكومة عن خطة جديدة لتطوير البنية التحتية في المناطق الريفية خلال السنوات الخمس القادمة
```
**8 errors uncorrected** - This appears to be a model initialization or first-line processing bug.

#### 2. Gender Agreement Errors (Systemic)
| Sentence | Error | Should Be |
|----------|-------|-----------|
| الطالبات المتفوقين حصلوا | المتفوقين حصلوا | المتفوقات حصلن |
| المهندسات الماهرات صمموا | صمموا | صممن |
| المدرسات الكفؤات يبدلوا | يبدلوا | يبذلن |
| الممرضات المتفانيين | المتفانيين | المتفانيات |
| الباحثات الجادون يعملوا | الجادون يعملوا | الجادات يعملن |
| المحاسبات الخبراء يدققوا | الخبراء يدققوا | الخبيرات يدققن |
| الموظفات المجتهدين | المجتهدين | المجتهدات |

#### 3. Number Agreement Errors (Systemic)
| Sentence | Error | Should Be |
|----------|-------|-----------|
| الموظفين الجدد يحتاجوا | يحتاجوا | يحتاجون |
| الشباب المتحمسين يبحثوا | يبحثوا | يبحثون |
| المهندسين الشباب يقدموا | يقدموا | يقدمون |
| المترجمين المحترفون يجيدون | المترجمين | المترجمون |

#### 4. Scattered Taa Marbuta Still Missed
- الوزاره (sentence 4)
- الفتره (sentence 4)
- المكتبه الرقميه (sentence 5)
- تربيه ابنائهن تربيه سليمه (sentence 17)
- ساعات طويله (sentence 28)
- And ~20 more instances

#### 5. Other Persistent Errors
| Error | Should Be | Type |
|-------|-----------|------|
| الطفل الصغيرة | الطفل الصغير | Gender (noun-adj) |
| القصة الذي | القصة التي | Relative pronoun gender |
| المدينة الكبير | المدينة الكبيرة | Gender (noun-adj) |
| وإثارة على البيئة | وآثاره على البيئة | Wrong word |
| نضام غدائى | نظام غذائي | ض/ظ + spelling |
| دراتها | ذاتها | ذ/د confusion |

---

## Error Category Analysis

### High Success Categories (>70%)
1. **Letter Confusion (ذ/د, ض/ظ)** - Model handles these well
2. **Common Hamza Patterns** - أعلنت, إلى, أكثر
3. **Word-final Taa Marbuta** - Most ه→ة at word end

### Medium Success Categories (40-70%)
1. **Mid-word Hamza** - Inconsistent
2. **Alif Maqsura** - Sometimes catches ى/ي
3. **Less Common Taa Marbuta** - Misses some patterns

### Low Success Categories (<30%)
1. **Gender Agreement** - Almost never corrected
2. **Number Agreement** - Rarely corrected
3. **Relative Pronoun Gender** - الذي/التي confusion
4. **Complex Morphology** - Verb conjugation for gender

---

## Root Cause Analysis

### Why Grammar Agreement Fails

1. **Training Data Bias**: QALB dataset has mostly orthographic errors, few agreement errors
2. **Synthetic Data Pattern**: Our synthetic corruption focused on letter-level changes, not grammatical
3. **Model Capacity**: 124M parameters may be insufficient for morphosyntactic agreement
4. **No Explicit Grammar**: Model learns patterns, not Arabic grammar rules

### Why First Sentence Fails

Hypotheses:
1. **Cold Start**: First sequence through model behaves differently
2. **Tokenization Edge Case**: BOS token interaction
3. **Attention Pattern**: First position gets less context
4. **Bug in Processing**: Line-by-line code may have edge case

### Why Some Taa Marbuta Missed

1. **Context Dependency**: Model may need more context for some words
2. **Frequency in Training**: Less common words may not have enough examples
3. **Position Effects**: Errors in middle of long sentences harder to catch

---

## Comparison: Model Strengths vs Weaknesses

```
STRONG                              WEAK
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Orthographic Errors                 Grammatical Agreement
  ├─ Taa Marbuta (ه→ة)               ├─ Gender (المتفوقين→المتفوقات)
  ├─ Hamza (ا→أ/إ)                   ├─ Number (يحتاجوا→يحتاجون)
  ├─ Letter confusion (ذ/د)          └─ Verb conjugation
  └─ Common spelling
                                    Relative Pronouns
                                      └─ الذي/التي for feminine

Single Word Fixes                   Multi-word Dependencies
  └─ Independent corrections          └─ Agreement across words

High-frequency Patterns             Low-frequency Patterns
  └─ Common words                     └─ Rare vocabulary
```

---

## Recommendations for Improvement

### Short Term (Data)
1. **Add Agreement Data**: Create synthetic data with gender/number agreement errors
2. **Increase QALB Weight**: More real data exposure
3. **Add Grammar Examples**: Curated agreement correction pairs

### Medium Term (Architecture)
1. **Grammar-Aware LoRA**: Fine-tune specifically on agreement errors
2. **Two-Stage Pipeline**: Orthographic → Grammar correction
3. **Rule Post-Processing**: Add rule-based agreement fixes

### Long Term (Model)
1. **Larger Model**: 300M+ for complex grammar
2. **Multi-Task Learning**: POS tagging + GEC jointly
3. **Arabic Morphology Integration**: Use morphological analyzers

---

## Benchmark Implications

### Current State
- **78.84% F0.5 on QALB-2014** (orthographic-heavy benchmark)
- **~50-60% on grammar-heavy text** (this test)

### Gap to SOTA (82.63%)
The 3.79 point gap is likely due to:
1. Grammar agreement errors (major)
2. Inconsistent orthographic correction (minor)
3. Edge cases like first sentence (minor)

### To Beat SOTA
Must address grammar agreement - this is the biggest lever remaining.

---

## Conclusion

The Nahawi model excels at orthographic correction (letter-level errors) but struggles with grammatical agreement (morphosyntactic errors). This aligns with the training approach which focused on synthetic orthographic errors + QALB data.

**Key Finding**: Two-pass correction provides marginal gains (~4% additional), suggesting the model's knowledge is largely exhausted after one pass. The remaining errors require fundamentally different training, not just more inference passes.

**Portfolio Implication**: The demo impressively corrects visible spelling errors (ه→ة, hamza, letter confusion) which are immediately recognizable improvements. The grammar gaps are less visible to non-linguists, making this a strong portfolio piece despite the technical limitations.

---

*Generated: January 2025*
*Test: 100 sentences, 2 passes*
*Model: Nahawi 124M + Punct-Aware LoRA (epoch_3)*
