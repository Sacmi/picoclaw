package agent

import (
	"context"
	"fmt"
	"strings"

	"github.com/sipeed/picoclaw/pkg/config"
	"github.com/sipeed/picoclaw/pkg/logger"
	"github.com/sipeed/picoclaw/pkg/providers"
	"github.com/sipeed/picoclaw/pkg/utils"
)

const topicNamePrompt = `Generate a very short topic label (2-4 words, max 25 chars) for a chat conversation based on the user's first message below. No emoji. Use the same language as the message. Be concise and descriptive. Return ONLY the topic name, nothing else.

%s`

const topicNameMaxRunes = 128

type topicNamer struct {
	provider providers.LLMProvider
	modelID  string
}

// NewTopicNamer creates a topicNamer from the topic_model config field.
// Returns nil if topic_model is not configured or the provider cannot be created.
func NewTopicNamer(cfg *config.Config) *topicNamer {
	if cfg == nil {
		return nil
	}
	modelName := strings.TrimSpace(cfg.Agents.Defaults.TopicModel)
	if modelName == "" {
		return nil
	}

	modelCfg, err := cfg.GetModelConfig(modelName)
	if err != nil {
		logger.WarnCF("agent", "topic_model not found in model_list", map[string]any{
			"topic_model": modelName,
			"error":       err.Error(),
		})
		return nil
	}

	provider, modelID, err := providers.CreateProviderFromConfig(modelCfg)
	if err != nil {
		logger.ErrorCF("agent", "Failed to create topic namer provider", map[string]any{
			"topic_model": modelName,
			"error":       err.Error(),
		})
		return nil
	}

	logger.InfoCF("agent", "Topic namer enabled", map[string]any{"model": modelID})
	return &topicNamer{provider: provider, modelID: modelID}
}

// GenerateName calls the configured model to produce a short topic name based on
// the first user message.
func (tn *topicNamer) GenerateName(ctx context.Context, userMessage, _ string) (string, error) {
	prompt := fmt.Sprintf(topicNamePrompt,
		utils.Truncate(userMessage, 500),
	)

	resp, err := tn.provider.Chat(ctx, []providers.Message{
		{Role: "user", Content: prompt},
	}, nil, tn.modelID, map[string]any{
		"temperature": 0.3,
		"max_tokens":  30,
	})
	if err != nil {
		return "", err
	}

	name := strings.TrimSpace(resp.Content)
	// Enforce Telegram's 128-character topic name limit.
	runes := []rune(name)
	if len(runes) > topicNameMaxRunes {
		name = string(runes[:topicNameMaxRunes-3]) + "..."
	}
	return name, nil
}
