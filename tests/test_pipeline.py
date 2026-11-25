import unittest
from unittest.mock import patch

from langchain_core.embeddings import Embeddings
from youtube_rag.generation import answer_question
from youtube_rag.retrieval import index_transcript


class TopicEmbeddings(Embeddings):
    """Deterministic test vectors for two topics, without model downloads."""

    def embed_query(self, text):
        return [float("motor" in text), float("garden" in text)]

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]


class PipelineTests(unittest.TestCase):
    @patch("youtube_rag.retrieval.HuggingFaceEmbeddings", return_value=TopicEmbeddings())
    def test_retrieval_isolated_by_transcript(self, _encoder):
        first = index_transcript(("garden flowers. " * 110) + ("motor bearings. " * 110), "first")
        second = index_transcript("garden soil.", "second")
        results = first.invoke("motor")
        self.assertIn("motor", results[0].page_content)
        self.assertEqual(len(results), 4)
        self.assertTrue(all(doc.metadata["source"] == "first" for doc in results))
        self.assertEqual([doc.page_content for doc in second.invoke("motor")], ["garden soil."])

    @patch("youtube_rag.generation._generator")
    def test_only_generated_answer_is_returned(self, make_generator):
        from langchain_core.documents import Document

        generator = make_generator.return_value
        generator.return_value = [{"generated_text": " Once per second. "}]
        docs = [Document(page_content="Measured once per second.", metadata={"passage": 2})]
        self.assertEqual(answer_question("How often?", docs), "Once per second.")
        messages = generator.tokenizer.apply_chat_template.call_args.args[0]
        self.assertIn("[Passage 2] Measured once per second.", messages[1]["content"])
        self.assertIn("How often?", messages[1]["content"])
        self.assertFalse(generator.call_args.kwargs["return_full_text"])
