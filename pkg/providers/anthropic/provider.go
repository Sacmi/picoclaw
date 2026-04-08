package anthropicprovider

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"

	"github.com/sipeed/picoclaw/pkg/providers/protocoltypes"
)

type (
	ToolCall               = protocoltypes.ToolCall
	FunctionCall           = protocoltypes.FunctionCall
	LLMResponse            = protocoltypes.LLMResponse
	UsageInfo              = protocoltypes.UsageInfo
	Message                = protocoltypes.Message
	ToolDefinition         = protocoltypes.ToolDefinition
	ToolFunctionDefinition = protocoltypes.ToolFunctionDefinition
)

const (
	defaultBaseURL    = "https://api.anthropic.com"
	claudeCodeVersion = "2.1.75"
)

var baseAnthropicOAuthBetas = []string{
	"claude-code-20250219",
	"oauth-2025-04-20",
	"fine-grained-tool-streaming-2025-05-14",
}

const anthropicOAuthIdentityPrompt = "You are Claude Code, Anthropic's official CLI for Claude."

const anthropicOAuthUserAgent = "claude-cli/" + claudeCodeVersion

var strippedAnthropicOAuthHeaders = []string{
	"X-Stainless-Lang",
	"X-Stainless-Package-Version",
	"X-Stainless-OS",
	"X-Stainless-Arch",
	"X-Stainless-Runtime",
	"X-Stainless-Runtime-Version",
	"X-Stainless-Retry-Count",
	"X-Stainless-Timeout",
}

type Provider struct {
	client      *anthropic.Client
	tokenSource func() (string, error)
	baseURL     string
}

// SupportsThinking implements providers.ThinkingCapable.
func (p *Provider) SupportsThinking() bool { return true }

func NewProvider(token string) *Provider {
	return NewProviderWithBaseURL(token, "")
}

func NewProviderWithBaseURL(token, apiBase string) *Provider {
	baseURL := normalizeBaseURL(apiBase)
	client := anthropic.NewClient(
		option.WithAuthToken(token),
		option.WithBaseURL(baseURL),
	)
	return &Provider{
		client:  &client,
		baseURL: baseURL,
	}
}

func NewProviderWithClient(client *anthropic.Client) *Provider {
	return &Provider{
		client:  client,
		baseURL: defaultBaseURL,
	}
}

func NewProviderWithTokenSource(token string, tokenSource func() (string, error)) *Provider {
	return NewProviderWithTokenSourceAndBaseURL(token, tokenSource, "")
}

func NewProviderWithTokenSourceAndBaseURL(token string, tokenSource func() (string, error), apiBase string) *Provider {
	p := NewProviderWithBaseURL(token, apiBase)
	p.tokenSource = tokenSource
	return p
}

func buildAnthropicOAuthRequestOptions(token, model string, options map[string]any) []option.RequestOption {
	betas := append([]string{}, baseAnthropicOAuthBetas...)
	if needsInterleavedThinkingBeta(model, options) {
		betas = append(betas, "interleaved-thinking-2025-05-14")
	}

	opts := []option.RequestOption{
		option.WithAuthToken(token),
		option.WithHeader("accept", "application/json"),
		option.WithHeader("anthropic-dangerous-direct-browser-access", "true"),
		option.WithHeader("anthropic-beta", strings.Join(betas, ",")),
		option.WithHeader("User-Agent", anthropicOAuthUserAgent),
		option.WithHeader("x-app", "cli"),
		option.WithMaxRetries(0),
	}
	for _, header := range strippedAnthropicOAuthHeaders {
		opts = append(opts, option.WithHeaderDel(header))
	}
	return opts
}

func (p *Provider) Chat(
	ctx context.Context,
	messages []Message,
	tools []ToolDefinition,
	model string,
	options map[string]any,
) (*LLMResponse, error) {
	params, err := buildParams(messages, tools, model, options, p.tokenSource != nil)
	if err != nil {
		return nil, err
	}

	var opts []option.RequestOption
	if p.tokenSource != nil {
		tok, err := p.tokenSource()
		if err != nil {
			return nil, fmt.Errorf("refreshing token: %w", err)
		}
		opts = append(opts, buildAnthropicOAuthRequestOptions(tok, model, options)...)
	}

	// OAuth/setup-tokens require streaming; API keys use non-streaming.
	if p.tokenSource != nil {
		return p.chatStreaming(ctx, params, tools, opts)
	}

	resp, err := p.client.Messages.New(ctx, params, opts...)
	if err != nil {
		return nil, fmt.Errorf("claude API call: %w", err)
	}

	return parseResponse(resp, tools), nil
}

func (p *Provider) chatStreaming(
	ctx context.Context,
	params anthropic.MessageNewParams,
	tools []ToolDefinition,
	opts []option.RequestOption,
) (*LLMResponse, error) {
	stream := p.client.Messages.NewStreaming(ctx, params, opts...)
	defer stream.Close()

	var msg anthropic.Message
	for stream.Next() {
		event := stream.Current()
		if err := msg.Accumulate(event); err != nil {
			return nil, fmt.Errorf("claude streaming accumulate: %w", err)
		}
	}
	if err := stream.Err(); err != nil {
		return nil, fmt.Errorf("claude API call: %w", err)
	}

	return parseResponse(&msg, tools), nil
}

func (p *Provider) GetDefaultModel() string {
	return "claude-sonnet-4.6"
}

func (p *Provider) BaseURL() string {
	return p.baseURL
}

func buildParams(
	messages []Message,
	tools []ToolDefinition,
	model string,
	options map[string]any,
	isOAuth bool,
) (anthropic.MessageNewParams, error) {
	var system []anthropic.TextBlockParam
	var anthropicMessages []anthropic.MessageParam
	var pendingToolUseIDs map[string]struct{}

	for _, msg := range messages {
		switch msg.Role {
		case "system":
			// Prefer structured SystemParts for per-block cache_control.
			// This enables LLM-side KV cache reuse: the static block's prefix
			// hash stays stable across requests while dynamic parts change freely.
			if len(msg.SystemParts) > 0 {
				for _, part := range msg.SystemParts {
					block := anthropic.TextBlockParam{Text: part.Text}
					if part.CacheControl != nil && part.CacheControl.Type == "ephemeral" {
						block.CacheControl = anthropic.NewCacheControlEphemeralParam()
					}
					system = append(system, block)
				}
			} else {
				system = append(system, anthropic.TextBlockParam{Text: msg.Content})
			}
		case "user":
			if msg.ToolCallID != "" {
				if _, ok := pendingToolUseIDs[msg.ToolCallID]; !ok {
					continue
				}
				anthropicMessages = append(anthropicMessages,
					anthropic.NewUserMessage(anthropic.NewToolResultBlock(msg.ToolCallID, msg.Content, false)),
				)
				delete(pendingToolUseIDs, msg.ToolCallID)
			} else {
				pendingToolUseIDs = nil
				anthropicMessages = append(anthropicMessages,
					anthropic.NewUserMessage(anthropic.NewTextBlock(msg.Content)),
				)
			}
		case "assistant":
			if len(msg.ToolCalls) > 0 {
				var blocks []anthropic.ContentBlockParamUnion
				emittedToolUseIDs := make(map[string]struct{}, len(msg.ToolCalls))
				if msg.Content != "" {
					blocks = append(blocks, anthropic.NewTextBlock(msg.Content))
				}
				for _, tc := range msg.ToolCalls {
					// Skip tool calls with empty names to avoid API errors
					if tc.Name == "" || tc.ID == "" {
						continue
					}
					name := tc.Name
					if isOAuth {
						name = toClaudeCodeName(name)
					}
					args := tc.Arguments
					if args == nil && tc.Function != nil && tc.Function.Arguments != "" {
						if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
							args = map[string]any{}
						}
					}
					if args == nil {
						args = map[string]any{}
					}
					blocks = append(blocks, anthropic.NewToolUseBlock(tc.ID, args, name))
					emittedToolUseIDs[tc.ID] = struct{}{}
				}
				if len(blocks) == 0 {
					pendingToolUseIDs = nil
					continue
				}
				anthropicMessages = append(anthropicMessages, anthropic.NewAssistantMessage(blocks...))
				if len(emittedToolUseIDs) > 0 {
					pendingToolUseIDs = emittedToolUseIDs
				} else {
					pendingToolUseIDs = nil
				}
			} else {
				pendingToolUseIDs = nil
				anthropicMessages = append(anthropicMessages,
					anthropic.NewAssistantMessage(anthropic.NewTextBlock(msg.Content)),
				)
			}
		case "tool":
			if _, ok := pendingToolUseIDs[msg.ToolCallID]; !ok {
				continue
			}
			anthropicMessages = append(anthropicMessages,
				anthropic.NewUserMessage(anthropic.NewToolResultBlock(msg.ToolCallID, msg.Content, false)),
			)
			delete(pendingToolUseIDs, msg.ToolCallID)
		}
	}

	maxTokens := int64(4096)
	if mt, ok := options["max_tokens"].(int); ok {
		maxTokens = int64(mt)
	}

	// Normalize model ID: Anthropic API uses hyphens (claude-sonnet-4-6),
	// but config may use dots (claude-sonnet-4.6).
	apiModel := strings.ReplaceAll(model, ".", "-")

	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(apiModel),
		Messages:  anthropicMessages,
		MaxTokens: maxTokens,
	}

	if isOAuth {
		system = append([]anthropic.TextBlockParam{{Text: anthropicOAuthIdentityPrompt}}, system...)
	}

	if len(system) > 0 {
		params.System = system
	}

	if temp, ok := options["temperature"].(float64); ok {
		params.Temperature = anthropic.Float(temp)
	}

	if len(tools) > 0 {
		params.Tools = translateTools(tools, isOAuth)
	}

	// Extended Thinking / Adaptive Thinking
	// The thinking_level value directly determines the API parameter format:
	//   "adaptive" → {thinking: {type: "adaptive"}} + output_config.effort
	//   "low/medium/high/xhigh" → {thinking: {type: "enabled", budget_tokens: N}}
	if level, ok := options["thinking_level"].(string); ok && level != "" && level != "off" {
		applyThinkingConfig(&params, level)
	}

	return params, nil
}

// applyThinkingConfig sets thinking parameters based on the level value.
// "adaptive" uses the adaptive thinking API (Claude 4.6+).
// All other levels use budget_tokens which is universally supported.
//
// Anthropic API constraint: temperature must not be set when thinking is enabled.
// budget_tokens must be strictly less than max_tokens.
func applyThinkingConfig(params *anthropic.MessageNewParams, level string) {
	// Anthropic API rejects requests with temperature set alongside thinking.
	// Reset to zero value (omitted from JSON serialization).
	if params.Temperature.Valid() {
		log.Printf("anthropic: temperature cleared because thinking is enabled (level=%s)", level)
	}
	params.Temperature = anthropic.MessageNewParams{}.Temperature

	if level == "adaptive" {
		adaptive := anthropic.NewThinkingConfigAdaptiveParam()
		params.Thinking = anthropic.ThinkingConfigParamUnion{OfAdaptive: &adaptive}
		params.OutputConfig = anthropic.OutputConfigParam{
			Effort: anthropic.OutputConfigEffortHigh,
		}
		return
	}

	budget := int64(levelToBudget(level))
	if budget <= 0 {
		return
	}

	// budget_tokens must be < max_tokens; clamp to respect user's max_tokens setting.
	if budget >= params.MaxTokens {
		log.Printf("anthropic: budget_tokens (%d) clamped to %d (max_tokens-1)", budget, params.MaxTokens-1)
		budget = params.MaxTokens - 1
	} else if budget > params.MaxTokens*80/100 {
		log.Printf("anthropic: thinking budget (%d) exceeds 80%% of max_tokens (%d), output may be truncated",
			budget, params.MaxTokens)
	}
	params.Thinking = anthropic.ThinkingConfigParamOfEnabled(budget)
}

// levelToBudget maps a thinking level to budget_tokens.
// Values are based on Anthropic's recommendations and community best practices:
//
//	low    =  4,096  — simple reasoning, quick debugging (Claude Code "think")
//	medium = 16,384  — Anthropic recommended sweet spot for most tasks
//	high   = 32,000  — complex architecture, deep analysis (diminishing returns above this)
//	xhigh  = 64,000  — extreme reasoning, research problems, benchmarks
//
// Note: For Claude 4.6+, prefer adaptive thinking over manual budget_tokens.
func levelToBudget(level string) int {
	switch level {
	case "low":
		return 4096
	case "medium":
		return 16384
	case "high":
		return 32000
	case "xhigh":
		return 64000
	default:
		return 0
	}
}

func translateTools(tools []ToolDefinition, isOAuth bool) []anthropic.ToolUnionParam {
	result := make([]anthropic.ToolUnionParam, 0, len(tools))
	for _, t := range tools {
		name := t.Function.Name
		if isOAuth {
			name = toClaudeCodeName(name)
		}
		tool := anthropic.ToolParam{
			Name: name,
			InputSchema: anthropic.ToolInputSchemaParam{
				Properties: t.Function.Parameters["properties"],
			},
		}
		if desc := t.Function.Description; desc != "" {
			tool.Description = anthropic.String(desc)
		}
		if req, ok := t.Function.Parameters["required"].([]any); ok {
			required := make([]string, 0, len(req))
			for _, r := range req {
				if s, ok := r.(string); ok {
					required = append(required, s)
				}
			}
			tool.InputSchema.Required = required
		}
		result = append(result, anthropic.ToolUnionParam{OfTool: &tool})
	}
	return result
}

func parseResponse(resp *anthropic.Message, tools []ToolDefinition) *LLMResponse {
	var content strings.Builder
	var reasoning strings.Builder
	var toolCalls []ToolCall

	for _, block := range resp.Content {
		switch block.Type {
		case "thinking":
			tb := block.AsThinking()
			reasoning.WriteString(tb.Thinking)
		case "text":
			tb := block.AsText()
			content.WriteString(tb.Text)
		case "tool_use":
			tu := block.AsToolUse()
			var args map[string]any
			if err := json.Unmarshal(tu.Input, &args); err != nil {
				log.Printf("anthropic: failed to decode tool call input for %q: %v", tu.Name, err)
				args = map[string]any{"raw": string(tu.Input)}
			}
			toolCalls = append(toolCalls, ToolCall{
				ID:        tu.ID,
				Name:      fromClaudeCodeName(tu.Name, tools),
				Arguments: args,
			})
		}
	}

	finishReason := "stop"
	switch resp.StopReason {
	case anthropic.StopReasonToolUse:
		finishReason = "tool_calls"
	case anthropic.StopReasonMaxTokens:
		finishReason = "length"
	case anthropic.StopReasonEndTurn:
		finishReason = "stop"
	}

	return &LLMResponse{
		Content:      content.String(),
		Reasoning:    reasoning.String(),
		ToolCalls:    toolCalls,
		FinishReason: finishReason,
		Usage: &UsageInfo{
			PromptTokens:     int(resp.Usage.InputTokens),
			CompletionTokens: int(resp.Usage.OutputTokens),
			TotalTokens:      int(resp.Usage.InputTokens + resp.Usage.OutputTokens),
		},
	}
}

func normalizeBaseURL(apiBase string) string {
	base := strings.TrimSpace(apiBase)
	if base == "" {
		return defaultBaseURL
	}

	base = strings.TrimRight(base, "/")
	if before, ok := strings.CutSuffix(base, "/v1"); ok {
		base = before
	}
	if base == "" {
		return defaultBaseURL
	}

	return base
}

var claudeCodeTools = []string{
	"Read",
	"Write",
	"Edit",
	"Bash",
	"Grep",
	"Glob",
	"AskUserQuestion",
	"EnterPlanMode",
	"ExitPlanMode",
	"KillShell",
	"NotebookEdit",
	"Skill",
	"Task",
	"TaskOutput",
	"TodoWrite",
	"WebFetch",
	"WebSearch",
}

var claudeCodeToolLookup = func() map[string]string {
	lookup := make(map[string]string, len(claudeCodeTools))
	for _, name := range claudeCodeTools {
		lookup[strings.ToLower(name)] = name
	}
	return lookup
}()

func toClaudeCodeName(name string) string {
	if canonical, ok := claudeCodeToolLookup[strings.ToLower(name)]; ok {
		return canonical
	}
	return name
}

func fromClaudeCodeName(name string, tools []ToolDefinition) string {
	lowerName := strings.ToLower(name)
	for _, tool := range tools {
		if strings.ToLower(tool.Function.Name) == lowerName {
			return tool.Function.Name
		}
	}
	return name
}

func needsInterleavedThinkingBeta(model string, options map[string]any) bool {
	if options != nil {
		if enabled, ok := options["interleaved_thinking"].(bool); ok {
			return enabled && !supportsAdaptiveThinking(model)
		}
	}
	return !supportsAdaptiveThinking(model)
}

func supportsAdaptiveThinking(model string) bool {
	normalized := strings.ToLower(strings.ReplaceAll(model, ".", "-"))
	return strings.Contains(normalized, "claude-sonnet-4-6") || strings.Contains(normalized, "claude-opus-4-6")
}
