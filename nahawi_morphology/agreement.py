"""
Arabic Agreement Checker

Checks and fixes agreement errors:
- Subject-verb agreement (gender, number)
- Noun-adjective agreement (gender, number, definiteness)
- Demonstrative-noun agreement
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
from .data import (
    ADJECTIVE_GENDER_PAIRS, ADJECTIVE_FEM_TO_MASC,
    IRREGULAR_VERBS, VOCAB_CACHE
)


@dataclass
class AgreementError:
    """Represents an agreement error."""
    error_type: str  # 'gender', 'number', 'definiteness'
    word1: str
    word2: str
    expected: str
    got: str
    suggestion: Optional[str] = None


class AgreementChecker:
    """
    Checks agreement between Arabic words.

    Usage:
        checker = AgreementChecker(analyzer)
        errors = checker.check("الطالبة", "المجتهد")
        # [AgreementError(type='gender', expected='المجتهدة', got='المجتهد')]
    """

    def __init__(self, analyzer):
        self.analyzer = analyzer
        self.adj_masc_to_fem = ADJECTIVE_GENDER_PAIRS
        self.adj_fem_to_masc = ADJECTIVE_FEM_TO_MASC
        self.irregular_verbs = IRREGULAR_VERBS

    def _is_adjective(self, word: str) -> bool:
        """Check if a word is likely an adjective."""
        # Strip definite article
        base = word[2:] if word.startswith('ال') else word

        # Check known adjective pairs
        if base in self.adj_masc_to_fem or base in self.adj_fem_to_masc:
            return True

        # Check with definite article
        if word in self.adj_masc_to_fem or word in self.adj_fem_to_masc:
            return True

        # Common adjective patterns
        # Active participle (فاعل): ends in consonant without ة for masc
        # Can follow a definite noun
        if word.startswith('ال') and not word.endswith(('ون', 'ين', 'ات')):
            # Could be an adjective following a noun
            return True

        return False

    def check(self, word1: str, word2: str) -> List[AgreementError]:
        """Check agreement between two words."""
        errors = []

        info1 = self.analyzer.analyze(word1)
        info2 = self.analyzer.analyze(word2)

        # Determine if word2 could be an adjective
        word2_is_adj = info2.pos == 'adj' or self._is_adjective(word2)

        # Noun/Subject + Adjective agreement
        if info1.pos in ('noun', 'adj') and word2_is_adj:
            errors.extend(self._check_noun_adj_agreement(word1, word2, info1, info2))

        # Subject + Verb agreement
        if info1.pos == 'noun' and info2.pos == 'verb':
            errors.extend(self._check_subject_verb_agreement(word1, word2, info1, info2))

        # Verb + Subject (VSO order)
        if info1.pos == 'verb' and info2.pos == 'noun':
            errors.extend(self._check_verb_subject_agreement(word1, word2, info1, info2))

        return errors

    def _check_noun_adj_agreement(self, noun: str, adj: str, info1, info2) -> List[AgreementError]:
        """Check noun-adjective agreement."""
        errors = []

        # Gender agreement
        if info1.gender != info2.gender and info1.gender != 'unknown' and info2.gender != 'unknown':
            suggestion = self._suggest_gender_fix(adj, info1.gender)
            errors.append(AgreementError(
                error_type='gender',
                word1=noun,
                word2=adj,
                expected=info1.gender,
                got=info2.gender,
                suggestion=suggestion
            ))

        # Number agreement (for human plurals)
        # Note: Non-human plurals take feminine singular adjectives
        if info1.number == 'plural' and info2.number == 'sing':
            # This might be correct for non-human plurals
            # For now, flag it
            suggestion = self._suggest_number_fix(adj, 'plural', info1.gender)
            if suggestion and suggestion != adj:
                errors.append(AgreementError(
                    error_type='number',
                    word1=noun,
                    word2=adj,
                    expected='plural',
                    got='sing',
                    suggestion=suggestion
                ))

        # Definiteness agreement
        if info1.definite != info2.definite:
            suggestion = self._suggest_definiteness_fix(adj, info1.definite)
            errors.append(AgreementError(
                error_type='definiteness',
                word1=noun,
                word2=adj,
                expected='definite' if info1.definite else 'indefinite',
                got='definite' if info2.definite else 'indefinite',
                suggestion=suggestion
            ))

        return errors

    def _check_subject_verb_agreement(self, subj: str, verb: str, info1, info2) -> List[AgreementError]:
        """Check subject-verb agreement (SVO order)."""
        errors = []

        # Gender agreement
        if info1.gender == 'fem' and info2.person and 'f' not in info2.person:
            suggestion = self._feminize_verb(verb, info2)
            errors.append(AgreementError(
                error_type='gender',
                word1=subj,
                word2=verb,
                expected='feminine verb',
                got='masculine verb',
                suggestion=suggestion
            ))
        elif info1.gender == 'masc' and info2.person and 'f' in info2.person and 'm' not in info2.person:
            suggestion = self._masculinize_verb(verb, info2)
            errors.append(AgreementError(
                error_type='gender',
                word1=subj,
                word2=verb,
                expected='masculine verb',
                got='feminine verb',
                suggestion=suggestion
            ))

        # Number agreement
        if info1.number == 'plural' and info2.person and 'p' not in info2.person:
            suggestion = self._pluralize_verb(verb, info2, info1.gender)
            if suggestion and suggestion != verb:
                errors.append(AgreementError(
                    error_type='number',
                    word1=subj,
                    word2=verb,
                    expected='plural verb',
                    got='singular verb',
                    suggestion=suggestion
                ))

        return errors

    def _check_verb_subject_agreement(self, verb: str, subj: str, info1, info2) -> List[AgreementError]:
        """Check verb-subject agreement (VSO order)."""
        errors = []

        # In VSO, verb agrees in gender but typically stays singular even for plural subjects
        if info2.gender == 'fem' and info1.person and 'f' not in info1.person:
            suggestion = self._feminize_verb(verb, info1)
            errors.append(AgreementError(
                error_type='gender',
                word1=verb,
                word2=subj,
                expected='feminine verb',
                got='masculine verb',
                suggestion=suggestion
            ))

        return errors

    def _suggest_gender_fix(self, adj: str, target_gender: str) -> Optional[str]:
        """Suggest the correct gender form of an adjective."""
        # Strip definite article for lookup
        base_adj = adj[2:] if adj.startswith('ال') else adj
        prefix = 'ال' if adj.startswith('ال') else ''

        if target_gender == 'fem':
            # Convert masculine to feminine
            if base_adj in self.adj_masc_to_fem:
                fem_form = self.adj_masc_to_fem[base_adj]
                return prefix + fem_form if not fem_form.startswith('ال') else fem_form
            # Generic: add ة
            if not base_adj.endswith('ة'):
                return prefix + base_adj + 'ة'
        else:
            # Convert feminine to masculine
            if base_adj in self.adj_fem_to_masc:
                masc_form = self.adj_fem_to_masc[base_adj]
                return prefix + masc_form if not masc_form.startswith('ال') else masc_form
            # Generic: remove ة
            if base_adj.endswith('ة'):
                return prefix + base_adj[:-1]

        return None

    def _suggest_number_fix(self, adj: str, target_number: str, gender: str) -> Optional[str]:
        """Suggest the correct number form of an adjective."""
        base_adj = adj[2:] if adj.startswith('ال') else adj
        prefix = 'ال' if adj.startswith('ال') else ''

        if target_number == 'plural':
            if gender == 'masc':
                # Sound masculine plural
                if base_adj.endswith('ة'):
                    base_adj = base_adj[:-1]
                return prefix + base_adj + 'ون'
            else:
                # Sound feminine plural
                if base_adj.endswith('ة'):
                    return prefix + base_adj[:-1] + 'ات'
                return prefix + base_adj + 'ات'

        return None

    def _suggest_definiteness_fix(self, adj: str, target_definite: bool) -> Optional[str]:
        """Suggest the correct definiteness form."""
        if target_definite:
            if not adj.startswith('ال'):
                return 'ال' + adj
        else:
            if adj.startswith('ال'):
                return adj[2:]
        return adj

    def _feminize_verb(self, verb: str, info) -> Optional[str]:
        """Convert a verb to its feminine form."""
        # Check irregular verbs
        for root, forms in self.irregular_verbs.items():
            for tense, persons in forms.items():
                for person, form in persons.items():
                    if verb == form and 'm' in person:
                        # Find the corresponding feminine form
                        fem_person = person.replace('m', 'f')
                        if fem_person in persons:
                            return persons[fem_person]

        # Check vocab cache
        if verb in VOCAB_CACHE:
            cached = VOCAB_CACHE[verb]
            if cached.get('pos') == 'verb':
                root = cached.get('root')
                tense = cached.get('tense')
                person = cached.get('person')
                if root and tense and person:
                    # Try to find feminine equivalent
                    fem_person = person.replace('m', 'f')
                    for v, c in VOCAB_CACHE.items():
                        if (c.get('root') == root and
                            c.get('tense') == tense and
                            c.get('person') == fem_person):
                            return v

        # Heuristic for past tense
        if info.tense == 'past':
            if not verb.endswith('ت') and not verb.endswith('وا'):
                return verb + 'ت'
            if verb.endswith('وا'):
                return verb[:-2] + 'ن'

        # Heuristic for present tense
        if info.tense == 'present':
            if verb.startswith('ي') and not verb.endswith('ن'):
                return 'ت' + verb[1:]

        return None

    def _masculinize_verb(self, verb: str, info) -> Optional[str]:
        """Convert a verb to its masculine form."""
        # Past tense: remove ت suffix
        if info.tense == 'past':
            if verb.endswith('ت') and len(verb) > 2:
                return verb[:-1]

        # Present tense: ت→ي prefix
        if info.tense == 'present':
            if verb.startswith('ت'):
                return 'ي' + verb[1:]

        return None

    def _pluralize_verb(self, verb: str, info, gender: str) -> Optional[str]:
        """Convert a verb to its plural form."""
        if info.tense == 'past':
            # Past plural: add وا
            if verb.endswith('ت'):
                if gender == 'fem':
                    return verb[:-1] + 'ن'
                else:
                    return verb[:-1] + 'وا'
            elif not verb.endswith('وا') and not verb.endswith('ن'):
                if gender == 'masc':
                    return verb + 'وا'
                else:
                    return verb + 'ن'

        if info.tense == 'present':
            # Present plural: add ون/ن
            if not verb.endswith('ون') and not verb.endswith('ن'):
                if gender == 'masc':
                    return verb + 'ون'
                else:
                    return verb + 'ن'

        return None

    def fix_sentence(self, sentence: str) -> str:
        """Fix agreement errors in a sentence."""
        words = sentence.split()
        result = []

        i = 0
        while i < len(words):
            word = words[i]

            # Look ahead for potential agreement issues
            if i + 1 < len(words):
                errors = self.check(word, words[i + 1])
                if errors:
                    # Apply the first fix
                    error = errors[0]
                    if error.suggestion:
                        result.append(word)
                        result.append(error.suggestion)
                        i += 2
                        continue

            result.append(word)
            i += 1

        return ' '.join(result)
