import pytest
from unittest.mock import patch, MagicMock
from lib.core.service.KnowledgeService import KnowledgeService


class TestKnowledgeService:
    def test_singleton_instance(self):
        """Test that KnowledgeService is a singleton."""
        instance1 = KnowledgeService()
        instance2 = KnowledgeService()
        assert instance1 is instance2

    @patch('lib.core.service.KnowledgeService.current_provider')
    def test_build_knowledge(self, mock_provider):
        """Test build_knowledge method."""
        mock_provider.embed.side_effect = lambda text: [float(len(text))]  # Mock embedding as length
        dataset = ["short", "longer text"]
        result = KnowledgeService.build_knowledge(dataset)
        expected = [("short", [5.0]), ("longer text", [11.0])]
        assert result == expected
        assert mock_provider.embed.call_count == 2

    @patch('lib.core.service.KnowledgeService.current_provider')
    @patch('lib.commons.MathUtils.MathUtils.cosine_similarity')
    def test_get_most_relevant_chunks(self, mock_similarity, mock_provider):
        """Test get_most_relevant_chunks method."""
        mock_provider.embed.return_value = [1.0, 0.0]
        knowledge = [("chunk1", [1.0, 0.0]), ("chunk2", [0.0, 1.0])]
        mock_similarity.side_effect = [0.8, 0.5]  # chunk1 more similar
        result = KnowledgeService.get_most_relevant_chunks("query", knowledge, top_n=1)
        assert len(result) == 1
        assert result[0][0] == "chunk1"
        assert result[0][1] == 0.8

    @patch('lib.core.service.KnowledgeService.current_provider')
    @patch('lib.commons.MathUtils.MathUtils.cosine_similarity')
    def test_get_best_matching_chunk(self, mock_similarity, mock_provider):
        """Test get_best_matching_chunk method."""
        mock_provider.embed.side_effect = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]  # query, chunk1, chunk2
        chunks = ["chunk1", "chunk2"]
        mock_similarity.side_effect = [0.9, 0.6]
        result = KnowledgeService.get_best_matching_chunk("query", chunks)
        assert result == {"match": "chunk1", "similarity": 0.9}

    @patch('lib.core.service.KnowledgeService.current_provider')
    @patch('lib.commons.MathUtils.MathUtils.cosine_similarity')
    def test_get_best_matching_chunk_no_chunks(self, mock_similarity, mock_provider):
        """Test get_best_matching_chunk with empty chunks."""
        result = KnowledgeService.get_best_matching_chunk("query", [])
        assert result is None
